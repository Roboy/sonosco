![# Sonosco](./imgs/sonosco_3.jpg)
<br>
<br>
<br>
<br>

Sonosco (from Lat. sonus - sound and nōscō - I know, recognize) 
is a library for training and deploying deep speech recognition models.

The goal of this project is to enable fast, repeatable and structured training of deep 
automatic speech recognition (ASR) models as well as providing a transcription server (REST API & frontend) to 
try out the trained models for transcription. <br>
Additionally, we provide interfaces to ROS in order to use it with 
the anthropomimetic robot [Roboy](https://roboy.org/).
<br>
<br>
<br>

___
### Installation

#### Via pip
The easiest way to use Sonosco's functionality is via pip:
```
pip install sonosco
```
**Note**: Sonosco requires Python 3.7 or higher. 

For reliability, we recommend using an environment virtualization tool, like virtualenv or conda.

#### For developers or trying out the transcription server
Clone the repository and install dependencies:
```
# Create a virtual python environment to not pollute the global setup
conda create -n 'sonosco' python=3.7

# activate the virtual environment
conda activate sonosco

# Clone the repo
git clone https://github.com/Roboy/sonosco.git

# Install normal requirements
pip install -r requirements.txt

# Link your local sonosco clone into your virtual environment
pip install .
```
Now you can check out some of the [Getting Started]() tutorials, to train a model or use 
the transcription server.
<br>
<br>
<br>
____________
### High Level Design


![# High-Level-Design](./imgs/high-level-design.svg)

The project is split into 4 parts that correlate with each other:

For data(-processing) scripts are provided to download and preprocess 
some publicly available datasets for speech recognition. Additionally, 
we provide scripts and functions to create manifest files 
(i.e. catalog files) for your own data and merge existing manifest files
into one.

This data or rather the manifest files can then be used to easily train and 
evaluate an ASR model. We provide some ASR model architectures, such as LAS, 
TDS and DeepSpeech2 but also individual pytorch models can be designed to be trained.

The trained model can then be used in a transcription server, that consists 
of a REST API as well as a simple Vue.js frontend to transcribe voice recorded 
by a microphone and compare the transcription results to other models (that can
be downloaded in our [Github](https://github.com/Roboy/sonosco) repository).

Further we provide example code, how to use different ASR models with ROS
and especially the Roboy ROS interfaces (i.e. topics & messages).

<br>
<br>

  
______
### Data (-processing)

##### Downloading publicly available datasets
We provide scripts to download and process the following publicly available datasets:
* [An4](http://www.speech.cs.cmu.edu/databases/an4/) - Alphanumeric database
* [Librispeech](http://www.openslr.org/12) - reading english books
* [TED-LIUM 3](https://lium.univ-lemans.fr/en/ted-lium3/) (ted3) - TED talks
* [Voxforge](http://www.voxforge.org/home/downloads)
* common voice (old version)

Simply run the respective scripts in `sonosco > datasets > download_datasets` with the
output_path flag and it will download and process the dataset. Further, it will create 
a manifest file for the dataset.

For example

```
python an4.py --target-dir temp/data/an4
```
<br>
<br>

##### Creating a manifest from your own data

If you want to create a manifest from your own data, order your files as follows:
```
data_directory    
└───txt
│   │   transcription01.txt
│   │   transcription02.txt
│   
└───wav
    │   audio01.wav
    │   audio02.wav
```
To create a manifest, run the `create_manifest.py` script with the data directory and an outputfile 
to automatically create a manifest file for your data.

For example:
```
python create_manifest.py --data_path path/to/data_directory --output-file temp/data/manifest.csv
```

<br>
<br>

##### Merging manifest files 

In order to merge multiple manifests into one, just specify a folder that contains all manifest 
files to be merged and run the ``` merge_manifest.py```.
This will look for all .csv files and merge the content together in the specified output-file.

For example:
```
python merge_manifest.py --merge-dir path/to/manifests_dir --output-path temp/manifests/merged_manifest.csv
```

<br>
<br>


___
### Model Training

One goal of this framework is to keep training as easy as possible and enable 
keeping track of already conducted experiments.

