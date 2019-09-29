.. _quick_start:

Use the Transcription Server
=============================

Dockerized inference server
------------------------------

Get the hold of our new fully trained models from the latest release! Try out the LAS model for the best performance.
Then specify the folder with the model to the runner script as shown underneath.

You can get the docker image from dockerhub under ``yuriyarabskyy/sonosco-inference:1.0``. Just run
``cd server && ./run.sh yuriyarabskyy/sonosco-inference:1.0`` to pull and start the server or optionally build your own image by executing the following commands.
::

    cd server

    # Build the docker image
    ./build.sh

    # Run the built image
    ./run.sh sonosco_server

You can also specify the path to your own models by writing
``./run.sh <image_name> <path/to/models>``.

Open ``http://localhost:5000`` in Chrome. You should be able to add models for performing
transcription by clicking on the plus button. Once the models are added, record your own
voice by clicking on the record button. You can replay and transcribe with the
corresponding buttons.

You can get pretrained models from the `release tab <https://github.com/Roboy/sonosco/releases>`_ in this repository.

To learn more see :ref:`Transcription Server <trans_server>`