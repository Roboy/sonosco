.. _trans_server:

Transcription Server
=====================

In the picture below, you see the GUI we created for our transcription server.
To start this server, follow the :ref:`Quick Start <quick_start>` .

.. image:: imgs/transcription_server.png

By pressing on the red plus button, one can add different models, that can be specified in a respective config file.<br>
Then you can record your voice, by clicking on the 'RECORD' button, listen to it by pressing the 'PLAY' button
and finally transcribing it with 'TRANSCRIBE'.

.. image:: imgs/transcription.png

The transcription of both models is displayed in the respective block. Further,
a popup shows up, where you are asked to correct the transcription. When you click on 'IMPROVE',
the audio and the respective transcription are saves to ``~/.sonosco/audio_data/web_collected/`` .
If one uses the 'Comparison' toggle, the transcriptions are additionally compared to the corrected transcription.

How to use the transcription server without docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Of course you can also use the transcription server without docker.
To do so, you first need to specify the model you want to use with its inference code in the ``model_loader.py``, this file can be
found at ``server > model_loader.py``.

For the LAS model, this looks like this:
::
    from sonosco.inference.las_inference import LasInference

    model_id_to_inference = {
        "las": LasInference}

Afterwards, just specify the model, with the path to the checkpoint file in the ``server > config.yaml``:
::
    models:
        -  id: 'las'
           path: '~/pretrained/las.pt'
           decoder: 'greedy'
           name: 'LAS'

The next thing to do is build the frontend, for this, navigate to ``server > frontend`` and run ``npm run build``.

When this is finished, start the transcription server by running ``python app.py`` in the ``server`` directory.
Open ``http://localhost:5000`` and use your model.


How to use your own model
^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use your own model with the model trainer, additionally to the serialization guide, you need to implement an inference snippet.
For this, simply follow the `example <https://github.com/Roboy/sonosco/tree/master/sonosco/inference>`_ of the other models.
And then specify your model in the ``../sonosco/server/model_loader.py`` script. (Have a look at the `repo <https://github.com/Roboy/sonosco/blob/master/server/model_loader.py>`_ )
