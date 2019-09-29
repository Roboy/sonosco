.. _training_overview:

Model Training
================

One goal of this framework is to keep training as easy as possible and enable
keeping track of already conducted experiments.

Analysis Object Model
----------------------

For model training, there are multiple objects that interact with each other.

.. image:: imgs/aom.svg

For Model training, one can define different metrics, that get evaluated during the training
process. These metrics get evaluated at specified steps during an epoch and during
validation.
Sonosco provides different metrics already, such as Word Error Rate or Character Error Rate.
But additional metrics can be created in a similar scheme.
See :ref:`Metrics <metrics>` .

Additionally, callbacks can be defined. A Callback is an arbitrary code that can be executed during
training. Sonosco provides for example different Callbacks, such as Learning Rate Reduction,
ModelSerialisationCallback, TensorboardCallback, ... 
Custom Callbacks can be defined following the examples. See :ref:`Callbacks <callbacks>` .

Most importantly, a model needs to be defined. The model is basically any torch module. For
(de-) serialisation, this model needs to conform to the :ref:`Serialisation Guide <serialisation>` .
Sonosco provides already existing model architectures that can be simply imported, such as
Listen Attend Spell, Sequence to Sequence with Time-depth Separable Convolutions and DeepSpeech2. 
See :ref:`Models <models>` .

We created a specific AudioDataset Class that is based on the pytorch Dataset class.
This AudioDataset requires an AudioDataProcessor in order to process the specified manifest file.
Further we created a special AudioDataLoader based on pytorch's Dataloader class, that
takes the AudioDataset and provides the data in batches to the model training.

Metrics, Callbacks, the Model and the AudioDataLoader are then provided to the ModelTrainer.
This ModelTrainer takes care of the training process. See :ref:`Train your first model <start_training>` .

The ModelTrainer can then be registered to the Experiment, that takes care of provenance.
I.e. when starting the training, all your code is time_stamped and saved in a separate directory,
so you can always repeat the same experiment. Additionally, the serialized model and modeltrainer,
logs and tensorboard logs are saved in this folder.

Further, a Serializer needs to be provided to the Experiment. This object can serialize any
arbitrary class with its parameters, that can then be deserialized using the Deserializer.
When the ``Experiment.stop()`` method is called, the model and the ModelTrainer get serialized,
so that you can simply continue the training, with all current parameters (such as epoch steps,...)
when deserializing the ModelTrainer and continuing training.
