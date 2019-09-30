Model Evaluation
=================

We provide model evaluation using the bootstrapping method. So the samples that the model is evaluated on are randomly sampled from
the testset with replacement.

You can pass the :ref:`Metrics <metrics>` that were also used for model training and evaluate your model on that.
You only need to specify:

 * model - i.e. a torch nn.module
 * dataloader - the test dataloader
 * bootstrap size - number of samples in contained in one bootstrap
 * num bootstraps - number of bootstraps to compute
 * torch device
 * metrics - List of metrics

The model evaluator will log its results in tensorboard and store it as a json file in the logs folder of the experiment.
With this method, one can evaluate the mean and variance per metric of the model.

To use the model trainer, just do: ``from sonosco.training import ModelEvaluator``