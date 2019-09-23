#!/usr/bin/env bash

if [ -z "$2" ]
  then
    echo "Using sonosco_server.."
    image="sonosco_server"
  else
    image=$2
fi

echo "Running docker image $image"

docker run -v $1:/work/pretrained_host -p 5000:5000 -p 8888:8888 -t $image
