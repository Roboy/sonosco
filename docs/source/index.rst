.. sonosco documentation master file, created by
   sphinx-quickstart on Wed Sep 25 19:18:50 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: imgs/sonosco_3.jpg

Sonosco (from Lat. sonus - sound and nōscō - I know, recognize)
is a library for training and deploying deep speech recognition models.

The goal of this project is to enable fast, repeatable and structured training of deep
automatic speech recognition (ASR) models as well as providing a transcription server (REST API & frontend) to
try out the trained models for transcription. 
Additionally, we provide interfaces to ROS in order to use it with
the anthropomimetic robot `Roboy <https://roboy.org/>`_.

The following diagram shows a brief overview of the functionalities of sonosco:

.. image:: imgs/high-level-design.svg

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
be downloaded in our `Github <https://github.com/Roboy/sonosco/releases>`_ repository).

Further we provide example code, how to use different ASR models with ROS
and especially the Roboy ROS interfaces (i.e. topics & messages).


Contents:
=========
.. _start:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: Installation and Getting Started

  start/*

.. _data:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: Data (-processing)

  data/*

.. _training:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: Model Training and Evaluation

  training/*

.. _server:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: Transcription Server

  server/*

.. _ros:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: ROS

  ros/*

.. _ackn:
.. toctree::
  :maxdepth: 1
  :glob:
  :caption: Acknowledgements

  acknowledgements/*