import logging
import sys

from .abstract_callback import AbstractCallback


LOGGER = logging.getLogger(__name__)


class EarlyStopping(AbstractCallback):
    """
    Early Stopping to terminate training early if the monitored metric did not improve
    over a number of epochs.
    Args:
        monitor (string): name of the relevant loss or metric (usually 'val_loss')
        min_delta (float): minimum change in monitored metric to qualify as an improvement
        patience (int): number of epochs to wait for an improvement before terminating the training
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.last_best = sys.float_info.max
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, epoch, step, performance_measures, context):

        if step != len(context.train_data_loader) - 1:  # only continue at end of epoch
            return

        if self.monitor not in performance_measures:
            return

        current_loss = performance_measures[self.monitor]
        if (self.last_best - current_loss) >= self.min_delta:
            self.last_best = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            context._stop_training = True   # make ModelTrainer stop
            LOGGER.info(f"Early stopping after epoch {epoch}")
