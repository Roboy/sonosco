# Roboy Sonosco
Roboy Sonosco (from Lat. sonus - sound and nōscō - I know, recognize) - a library for Speech Recognition based on Deep Learning models

## Installation

The supported OS is Ubuntu 18.04 LTS (however, it should work fine on other distributions).
Supported Python version is 3.6+.
Supported CUDA version is 10.0.
Supported PyTorch version is 1.0.

---

Install CUDA 10.0 from [NVIDIA website](https://developer.nvidia.com/cuda-10.0-download-archive). Make sure that your local gcc, g++, cmake versions are not older than the ones used to compile your OS kernel.

You will need to download the latest [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) for CUDA 10.0. 
Unzip it:
```
tar -xzvf cudnn-9.0-linux-x64-v7.tgz
```
Run
```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
---

**All of the following steps you may perform inside [Anaconda](https://www.anaconda.com/) or [virtualenv](https://virtualenv.pypa.io/en/latest/)**

Install [PyTorch](https://pytorch.org/get-started/locally/). For your particular configuration, you may want to build it from the [sources](https://github.com/pytorch/pytorch).

Install SeanNaren's fork for Warp-CTC bindings. **Deprecated**: will be updated to use [built-in](https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss) functions.
```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
```

Install pytorch audio:
```
sudo apt-get install sox libsox-dev libsox-fmt-all
git clone https://github.com/pytorch/audio.git
cd audio && python setup.py install
```

If you want decoding to support beam search with an optional language model, install [ctcdecode](https://github.com/parlance/ctcdecode):
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

Clone this repo and run this within the repo:
```
pip install -r requirements.txt
```

### Mixed Precision
If you want to use mixed precision training, you have to install [NVIDIA Apex](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/):
```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

## Usage

### Dataset

To create a dataset you must create a CSV manifest file containing the locations of the training data. This has to be in the format of:
```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
There is an example in examples directory.

### Training, Testing and Inference

Fundamentally, you can run the scripts the same way:
```
python3 train.py --config /path/to/config/file.yaml
python3 test.py --config /path/to/config/file.yaml
python3 infer.py --config /path/to/config/file.yaml
```
The scripts are initialised via configuration files.

#### Configuration

Configuration file contains arguments for ModelWrapper initialisation as well as extra parameters. Like this:
```
train:
  ...
  log-dir: 'logs' # Location for log files
  def-dir: 'examples/checkpoints/', # Default location to save/load models
  model-name: 'asr_final.pth' # File name to save the best model
  sample-rate: 16000 # Sample rate
  window: 'hamming' # Window type for spectrogram generation
  batch-size: 32 # Batch size for training
  checkpoint: True # Enables checkpoint saving of model
  ...
```
More configuration examples with descriptions you may find in the config directory.

## Acknowledgements

This project is partially based on SeanNaren's [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) repository.
