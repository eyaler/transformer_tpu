# The rough guide to running transformer on TPU (TF 1.9.0rc2, T2T 1.6.6, 2018-07-09)
see also:

https://cloud.google.com/tpu/docs/custom-setup

https://github.com/tensorflow/tensor2tensor/blob/master/docs/cloud_tpu.md

https://github.com/tensorflow/tensor2tensor/blob/master/docs/walkthrough.md

```
# create storage bucket (one-time)
gsutil mb gs://myproject-storage

# install gcloud: https://cloud.google.com/sdk/install
# or update: gcloud components update

# create VM with disk (disk needed on one VM for data generation)
gcloud compute instances create myuser-vm2 --machine-type=n1-standard-1 --image-project=ml-images --image-family=tf-1-9 --scopes=cloud-platform --create-disk size=30,type=pd-standard

# ssh to the VM
# you can find the key under IdentityFile here:
# cat ~/.ssh/config
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
export OUT_DIR=$GCS_BUCKET/t2t/training/transformer_vm2

# generate dataset
t2t-datagen --problem=translate_ende_wmt32k --data_dir=$DATA_DIR --tmp_dir=/mnt/disks/tmp

## train on new TPU

# fix trainer code:
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/bin/t2t_trainer.py
# change all os.getenv("USER") to os.uname()[1], to make sure each VM has its own TPU

# fix TPU code:
sudo vi /usr/local/lib/python3.5/dist-packages/tensor2tensor/utils/cloud_tpu.py
# change all 1-8 and 1.8 to 1-9 and 1.9 respectively
# add to create_tpu(cls) after --version=%s: --network=default (https://github.com/tensorflow/tensor2tensor/issues/924)

# create passphrase (https://github.com/tensorflow/tensor2tensor/issues/920)
gcloud compute ssh $HOSTNAME-vm
# press enter
exit

tmux
t2t-trainer --model=transformer --hparams_set=transformer_tpu --problem=translate_ende_wmt32k --train_steps=100000 --eval_steps=10 --local_eval_frequency=100 --data_dir=$DATA_DIR --output_dir=$OUT_DIR --cloud_tpu --cloud_delete_on_done --cloud_skip_confirmation
# note will fail if eval_steps is too high

# run tensorboard
# open console shell by browsing to: https://console.cloud.google.com/cloudshell
# in the shell:
tensorboard --logdir=gs://myproject-storage/t2t/training --port=8080
# click on "web preview" button

## test
echo -e 'Hello world\nGood world' > text.en
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de
BEAM_SIZE=4
ALPHA=0.6
t2t-decoder --data_dir=$DATA_DIR --problem=translate_ende_wmt32k --model=transformer --hparams_set=transformer_tpu --output_dir=$OUT_DIR --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" --decode_from_file=text.en --decode_to_file=translation.en
cat translation.en

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.en --reference=ref-translation.de
```
