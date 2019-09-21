import logging
import time

from typing import Dict
from sonosco.serialization import serializable
from ..abstract_callback import AbstractCallback, ModelTrainer

LOGGER = logging.getLogger(__name__)


@serializable
class EpochEstimationCallback(AbstractCallback):
    """
    Estimates time required for one epoch at given batch step.

    Args:
        evaluate_at_batch_step (int): step, at which epoch time is evaluated

    """
    evaluate_at_batch_step: int

    def __call__(self,
                 epoch: int,
                 step: int,
                 performance_measures: Dict,
                 context: ModelTrainer,
                 validation: bool = False) -> None:
        """
        Estimate time required an epoch of training.

        Args:
            epoch: epoch step
            step: step inside of the epoch
            performance_measures: performance measures dictionary
            context: model trainer
            validation: should validation dataloader be used for comparison

        """
        if step == 0:
            self.start = time.time()
            self.total_batch_size = len(context.train_data_loader)

        if step == self.evaluate_at_batch_step:
            self.end = time.time()
            delta = int((self.end - self.start))
            total_time = self.total_batch_size/step*delta
            LOGGER.info(f'Model will require ~ {int(total_time/3600)} hours {int(total_time%3600/60)} minutes and {int(total_time%3600 % 60)} seconds for one epoch.')
            [context.callbacks.remove(callback) for callback in context.callbacks if isinstance(callback, EpochEstimationCallback)]
