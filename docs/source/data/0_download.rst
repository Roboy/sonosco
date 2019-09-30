.. _download:

Downloading publicly available datasets
=================================================


We provide scripts to download and process the following publicly available datasets:

 * `An4 <http://www.speech.cs.cmu.edu/databases/an4/>`_ - Alphanumeric database
 * `Librispeech <http://www.openslr.org/12>`_ - reading english books
 * `TED-LIUM 3 <https://lium.univ-lemans.fr/en/ted-lium3/>`_ (ted3) - TED talks
 * `Voxforge <http://www.voxforge.org/home/downloads>`_
 * common voice (old version)

Simply run the respective scripts in ``sonosco > datasets > download_datasets`` with the
output_path flag and it will download and process the dataset. Further, it will create
a manifest file for the dataset.

For example
::
    python an4.py --target-dir temp/data/an4
