#!/bin/sh
IMAGE_FAMILY="pytorch-latest-cpu"
ZONE="us-east1-c"
INSTANCE_NAME="control-cpu"
# maybe change the number of cpus? - last number
INSTANCE_TYPE="n1-standard-4"
IMAGE_PROJECT="deeplearning-platform-release"
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=120GB