.. _callbacks:

Callbacks
===========

A callback is an arbitrary code that can be executed during training. 
When defining a new callback all you need to do is inherit from the AbstractCallback class:
::
    @serializable
    class AbstractCallback(ABC):
        """
        Interface that defines how callbacks must be specified.
        """

        @abstractmethod
        def __call__(self, epoch: int, step: int, performance_measures: dict, context: ModelTrainer) -> None:
            """
            Called after every batch by the ModelTrainer.

            Args:
                epoch (int): current epoch number
                step (int): current batch number
                performance_measures (dict): losses and metrics based on a running average
                context (ModelTrainer): reference to the calling ModelTrainer, allows to access members

            """
            pass

        def close(self) -> None:
            """
            Handle cleanup work if necessary. Will be called at the end of the last epoch.
            """
            pass

Each callback is called at every epoch step and receives the current epoch, the current step, all
performance measures , i.e. a list of all metrics and the context, which is the modeltrainer itself,
so you can access all parameters of the modeltrainer from within the callback.

Sonosco already provides a lot of callbacks:
 * Early Stopping: Early Stopping to terminate training early if the monitored metric did not improve
    over a number of epochs.
 * Gradient Collector: Collects the layer-wise gradient norms for each epoch.
 * History Recorder: Records all losses and metrics during training.

 * Stepwise Learning Rate Reduction: Reduces the learning rate of the optimizer every N epochs.
 * Scheduled Learning Rate Reduction: Reduces the learning rate of the optimizer for every scheduled epoch.
 * Model Checkpoint: Saves the model and optimizer state at the point with lowest validation error throughout training.

 Further, sonosco provides a couple of callbacks that store information for tensorboard:
 * TensorBoard Callback: Log all metrics in tensorboard.
 * Tb Text Comparison Callback: Perform inference on a tds model and compare the generated text with groundtruth and add it to tensorboard.
 * Tb Teacher Forcing Text Comparison Callback: Perform decoding using teacher forcing and compare predictions to groundtruth and visualize in tensorboard.
 * Las Text Comparison Callback: Perform inference on an las model and compare the generated text with groundtruth and add it to tensorboard.
