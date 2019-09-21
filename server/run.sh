#!/usr/bin/env bash
docker run -v $1:/work/pretrained_host  -p 5000:5000 -p 8888:8888 -it sonosco_server /bin/bash
