#!/usr/bin/env bash
docker run -v ../pretrained:/work/pretrained_host  -p 5000:5000 -p 8888:8888 -it ros1_roboy /bin/bash
