#!/usr/bin/env bash

if [ -z "$2" ]
  then
    echo "Using sonosco_server.."
    image_name = "sonosco_server"
  else
    image_name = $2
fi

docker run -v $1:/work/pretrained_host  -p 5000:5000 -p 8888:8888 -t $image_name
