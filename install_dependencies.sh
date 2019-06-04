#!/usr/bin/env bash

#This scripts assumes that you have a virtual env in ./venv, you can override this by ./install_dependencies.sh -p /some/other/path

set -e

# define arguments
for i in "$@"
do
case ${i} in
    -c=*|--cuda=*)
    CUDA="${i#*=}"
    shift # past argument=value
    ;;
    -a=*|--anaconda=*)
    ANACONDA="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--venv=*)
    VENV="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

VENV=${VENV:-./venv}

if [ -z ${ANACONDA+x} ] ; then
    conda activate ${ANACONDA}
elif [ -z ${VENV+x} ] ; then
    source ${VENV}/bin/activate
fi

#TODO: Infer this automatically
CUDA=${CUDA:-false}

pip install -r requirements.txt

git clone https://github.com/SeanNaren/warp-ctc.git
if [ "$CUDA" = false ] ; then
    # This works for mac, for other OSes remove '' after -i
    sed -i '' 's/option(WITH_OMP \"compile warp-ctc with openmp.\" ON)/option(WITH_OMP \"compile warp-ctc with openmp.\" ${CUDA_FOUND})/' warp-ctc/CMakeLists.txt
else
    export CUDA_HOME="/usr/local/cuda"
fi
cd warp-ctc; mkdir build; cd build; cmake ..; make
cd ../pytorch_binding && python setup.py install
cd ../..
rm -rf warp-ctc

git clone git@github.com:pytorch/audio.git
cd audio; MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install
cd ..
rm -rf audio

pip install -r post_requirements.txt