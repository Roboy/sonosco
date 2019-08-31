import logging
import time
import torch

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class EpochEstimationCallback(AbstractCallback):
    '''
    Estimates time required for one epoch at given batch step.
    Args:
        evaluate_at_batch_step: (int) - step, at which epoch time is evaluated.
    '''
    evaluate_at_batch_step: int

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):
        if step == 0:
            self.start = time.time()
            self.total_batch_size = len(context.train_data_loader)
        if step == self.evaluate_at_batch_step:
            self.end = time.time()
            delta = int((self.end - self.start))
            total_time = self.total_batch_size/step*delta
            LOGGER.info(f'Model will require ~ {int(total_time/3600)} hours {int(total_time%3600/60)} minutes and {int(total_time%3600 % 60)} seconds for one epoch.')
            [context.callbacks.remove(callback) for callback in context.callbacks if isinstance(callback, EpochEstimationCallback)]
