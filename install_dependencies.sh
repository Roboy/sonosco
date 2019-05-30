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
    -p=*|--python_path=*)
    PYTHON_HOME_PATH="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

PYTHON_HOME_PATH=${PYTHON_HOME_PATH:-./venv}
#TODO: Infer this automatically
CUDA=${CUDA:-false}
source ${PYTHON_HOME_PATH}/bin/activate

pip install -r requirements.txt

git clone https://github.com/SeanNaren/warp-ctc.git
if [ "$CUDA" = false ] ; then
    sed -i '' 's/option(WITH_OMP \"compile warp-ctc with openmp.\" ON)/option(WITH_OMP \"compile warp-ctc with openmp.\" ${CUDA_FOUND})/' warp-ctc/CMakeLists.txt
else
    export CUDA_HOME="/usr/local/cuda"
fi
cd warp-ctc; mkdir build; cd build; cmake ..; make
cd ../pytorch_binding && python setup.py install
cd ../..
rm -rf warp-ctc

pip install -r post_requirements.txt