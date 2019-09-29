.. _start_training:

Train your first model
========================

In the following we will give you a short tutorial how to train the Listen Attend Spell model.

First of all, it is required to have some data. For that simply download one of the publicly available datasets using one of our scripts, see the Data Section :ref:`download <download>` .

In the next step, create a ``config.yaml`` file, that contains the following:
::
    experiment:
      experiment_path: "path/to_directory/where/experiment/is_stored"
      experiment_name: 'getting_started_w_las'
      gloabl_seed: 1234

    data:
      train_manifest: "path/to/train_manifest.csv"
      val_manifest: "path/to/val_manifest.csv"
      test_manifest: "path/to/test_manifest.csv"
      batch_size: 4
      num_data_workers: 8

    training:
      max_epochs: 10
      learning_rate: 1.0e-3
      labels: "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
      test_step: 1
      checkpoint_path: 'path/to/checkpoint_dir_in_experiment'
      window_size: 0.02
      window_stride: 0.01
      window: 'hamming'
      sample_rate: 16000

    model:
      encoder:
        input_size: 161
        hidden_size: 256
        num_layers: 1
        dropout: 0.2
        bidirectional: True
        rnn_type: "lstm"

      decoder:
        embedding_dim: 128
        hidden_size: 512
        num_layers: 1
        bidirectional_encoder: True
        vocab_size: 28
        sos_id: 27
        eos_id: 26

Under 'experiment_path' all your code, logs and checkpoints will be saved during training.
Further a global seed per experiment can be set. 
Under 'data' the manifest files for the data need to be specified, together with batch size and
the amount of workers for the dataloader.
'training' contains the maximum epochs to be trained, the initial learning rate, the labels, i.e.
characters the model is trained on and test_step is the step in an epoch, where validation is performed.

'model' contains parameters for the model to be trained, in this case, the LAS model requires parameters
for its encoder and decoder.

Let's setup the logger:
::

    import logging
    reload(logging)
    logging.basicConfig(level=logging.INFO)


In order to start training, we first need to import all required functions:
::
    import torch
    from sonosco.common.path_utils import parse_yaml
    from sonosco.datasets import create_data_loaders

    from sonosco.training import Experiment, ModelTrainer
    from sonosco.serialization import Deserializer
    from sonosco.decoders import GreedyDecoder
    from sonosco.models import Seq2Seq

    from sonosco.training.metrics import word_error_rate
    from sonosco.training.metrics import character_error_rate
    from sonosco.training.losses import cross_entropy_loss

Let's load the created .yaml file and create the dataloaders:
::
    config = parse_yaml('path/to/config.yaml')
    device = torch.device("cpu")
    train_loader, val_loader, test_loader = create_data_loaders(**config['data'])

For the model trainer, we can create a dict, that is then just passed for initialization:
::
    training_args = {
        'loss': cross_entropy_loss,
        'epochs': config['training']["max_epochs"],
        'train_data_loader': train_loader,
        'val_data_loader': val_loader,
        'test_data_loader': test_loader,
        'lr': config['training']["learning_rate"],
        'custom_model_eval': True,
        'metrics': [word_error_rate, character_error_rate],
        'decoder': GreedyDecoder(config['training']['labels']),
        'device': device,
        'test_step': config['training']["test_step"]}

With the following code, you can now easily start training and continue it:
::
    experiment = Experiment.create(config, logging.getLogger())

    CONTINUE = False

    if not CONTINUE:
        model = Seq2Seq(config['model']["encoder"], config['model']["decoder"])
        trainer = ModelTrainer(model, **training_args)
    else:
        loader = Deserializer()
        trainer, config = loader.deserialize(ModelTrainer, config['training']["checkpoint_path"], {
                'train_data_loader': train_loader,'val_data_loader': val_loader, 'test_data_loader': test_loader,
        }, with_config=True)

    experiment.setup_model_trainer(trainer, checkpoints=True, tensorboard=True)

    try:
        if not CONTINUE:
            experiment.start()
        else:
            experiment.__trainer.continue_training()
    except KeyboardInterrupt:
        experiment.stop()

We now go through this snippet in detail:
First, we set up the experiment and set the bool ``CONTINUTE=FALSE`` so that we
start the training.
We setup the modeltrainer with the las model and all the parameters we specified in the ``training_args``` dictionary.

Now we register the modeltrainer to the experiment and start it.

The try-except block catches keyboard interuptions, where the experiment will then save the model checkpoint aswell as the model trainer.
This serialized model trainer can then be used to continue training, just by setting the ``CONTINUE=True`` and rerunning the script.
What happens now is, that the modeltrainer, that is saved at the path, specified in the config file, is deserialized and continues training.

That's it !
You successfully train an LAS model.

For a more detailed description of each component, have a look at general description :ref:`training <training_overview>` of the model training process and its components.