# The rough guide to running transformer on TPU (TF 1.9.0, T2T 1.6.6, 2018-07-09)
see also:

https://cloud.google.com/tpu/docs/custom-setup

https://github.com/tensorflow/tensor2tensor/blob/master/docs/cloud_tpu.md

https://github.com/tensorflow/tensor2tensor/blob/master/docs/walkthrough.md

```
# install gcloud: https://cloud.google.com/sdk/install
# or update:
gcloud components update

# configure local machine
gcloud auth application-default login
gcloud config set project myproject
gcloud config set compute/zone us-central1-f

# create storage bucket (one-time)
gsutil mb gs://myproject-storage

# create VM with enough memory for eval and enough disk to prepare data
gcloud compute instances create myuser-vm2 --machine-type=n1-standard-2 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform --create-disk size=30,type=pd-standard

# ssh to the VM
# you can find the key under IdentityFile here:
cat ~/.ssh/config
# your user is your local user

# configure VM
sudo apt-get update && sudo apt-get --only-upgrade install kubectl google-cloud-sdk google-cloud-sdk-app-engine-grpc google-cloud-sdk-pubsub-emulator google-cloud-sdk-app-engine-go google-cloud-sdk-datastore-emulator google-cloud-sdk-app-engine-python google-cloud-sdk-cbt google-cloud-sdk-bigtable-emulator google-cloud-sdk-app-engine-python-extras google-cloud-sdk-datalab google-cloud-sdk-app-engine-java
gcloud config set project myproject
gcloud config set compute/zone us-central1-f

# connect disk
lsblk
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p /mnt/disks/tmp
sudo mount -o discard,defaults /dev/sdb /mnt/disks/tmp
sudo chmod a+w /mnt/disks/tmp

# define variables
export GCS_BUCKET=gs://myproject-storage
export DATA_DIR=$GCS_BUCKET/t2t/data
export OUT_DIR=$GCS_BUCKET/t2t/training/transformer_$HOSTNAME
export BEAM_SIZE=4
export ALPHA=0.6

# prepare dataset
t2t-datagen --problem=translate_ende_wmt32k --data_dir=$DATA_DIR --tmp_dir=/mnt/disks/tmp

## train on new TPU

# fix trainer code:
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/bin/t2t_trainer.py
# change all os.getenv("USER") to os.uname()[1], to make sure each VM has its own TPU

# fix TPU code:
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/utils/cloud_tpu.py
# change all 1-8 and 1.8 to 1-9 and 1.9 respectively
# in create_tpu(cls) after --version=%s add: --network=default (https://github.com/tensorflow/tensor2tensor/issues/924)

# create passphrase (https://github.com/tensorflow/tensor2tensor/issues/920)
gcloud compute ssh $HOSTNAME-vm
# press enter twice
# resource not found error is expected

tmux
t2t-trainer --model=transformer --hparams_set=transformer_tpu --problem=translate_ende_wmt32k --train_steps=250000 --eval_steps=10 --local_eval_frequency=100 --data_dir=$DATA_DIR --output_dir=$OUT_DIR --cloud_tpu --cloud_delete_on_done --cloud_skip_confirmation
# note: will fail if eval_steps is too high

# running tensorboard (solution that works also for windows)
# note: don't try to run it on gcp console shell. it is unusably slow
# following https://stackoverflow.com/a/42049234/664456
# open port in firewall (first time only):
gcloud compute firewall-rules create tensorboard-port --allow tcp:6006
# open a new terminal on one of your VM's
tmux
tensorboard --logdir=gs://myproject-storage/t2t/training
# in your local browser open: IP_OF_VM:6006

## test
echo -e 'Hello world\nGood world' > text.en
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de
t2t-decoder --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --model=transformer --hparams_set=transformer_tpu --output_dir=$OUT_DIR --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" --decode_from_file=text.en --decode_to_file=translation.de
cat translation.de

# evaluate. make sure to use UNTOKENIZED data (and see also https://github.com/tensorflow/tensor2tensor/issues/317)
wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.en
wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.de
t2t-decoder --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --model=transformer --hparams_set=transformer_tpu --output_dir=$OUT_DIR --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" --decode_from_file=newstest2014.en --decode_to_file=translation.de
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.de --reference=newstest2014.de
# for 250,000 epochs i get:
BLEU_uncased =  26.60
BLEU_cased =  26.11
# which is comparable to 27.3 of arxiv.org/abs/1706.03762
# as well as 28 for 300,000 epochs as reported in https://github.com/tensorflow/tensor2tensor
# differences are expected due to TPU and batch size

# you will get the same numbers with sacreBLEU
sudo pip3 install sacrebleu
sacrebleu -t wmt14 -l en-de --echo src > wmt14-en-de.src
t2t-decoder --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --model=transformer --hparams_set=transformer_tpu --output_dir=$OUT_DIR --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" --decode_from_file=wmt14-en-de.src --decode_to_file=translation.de
cat translation.de | sacrebleu -t wmt14/full -l en-de -tok intl -lc
#BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.2.10 = 26.60 58.5/32.3/20.2/13.2 (BP = 1.000 ratio = 1.026 hyp_len = 66350 ref_len = 64676)
cat translation.de | sacrebleu -t wmt14/full -l en-de -tok intl
# BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.2.10 = 26.11 57.4/31.7/19.8/12.9 (BP = 1.000 ratio = 1.026 hyp_len = 66350 ref_len = 64676)

# (optional) delete checkpoint files to restart training
gsutil rm -r $OUT_DIR

# (optional) delete failed instances to cleanup or to fix errors (e.g. Deadline Exceeded due to trying running concurrent stuff on the same TPU)
gcloud compute instances delete $HOSTNAME-vm --quiet
gcloud compute tpus delete $HOSTNAME-tpu --quiet

## (optional) running faster
# https://cloud.google.com/tpu/docs/performance-guide
# https://cloud.google.com/tpu/docs/troubleshooting
# https://arxiv.org/abs/1804.00247
# https://arxiv.org/abs/1806.00187

# increase batch size
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/models/transformer.py
# in update_hparams_for_tpu(hparams) change: batch_size = 18432

# change activations and weights to float16
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/layers/common_layers.py
# in basic_params1() change: activation_dtype="bfloat16" and weight_dtype="bfloat16"
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/layers/common_attention.py
# in get_timing_signal_1d(...) change: return tf.cast(signal, tf.bfloat16)
# (https://github.com/tensorflow/tensor2tensor/issues/932)

# note: disable eval to be quicker and prevent memory error
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/bin/t2t_trainer.py
# chenge: "schedule", "train"

# delete checkpoints and instances as described above and train
```
