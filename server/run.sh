#!/usr/bin/env bash

if [ -z "$2" ]
  then
    mkdir pretrained
    pretrained_path="$(pwd)/pretrained"
    echo "Downloading las model.."
    wget -P $pretrained_path https://github.com/Roboy/sonosco/releases/download/v1.0-beta/las_model_5.pt
  else
    pretrained_path=$2
fi

echo "Running docker image $image"

docker run -v $pretrained_path:/work/pretrained_host -p 5000:5000 -p 8888:8888 -t $1
