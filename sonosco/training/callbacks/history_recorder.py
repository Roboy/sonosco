import logging
import torch

from typing import Dict
from collections import defaultdict
from ..abstract_callback import AbstractCallback, ModelTrainer


LOGGER = logging.getLogger(__name__)


class HistoryRecorder(AbstractCallback):
    """
    Records all losses and metrics during training.
    """

    def __init__(self, epoch_steps):
        self.history = defaultdict(list)
        self._epoch_steps = epoch_steps

    def __call__(self,
                 epoch: int,
                 step: int,
                 performance_measures: Dict,
                 context: ModelTrainer,
                 validation: bool = False) -> None:
        """
        Execute history recording.

        Args:
            epoch: epoch step
            step: step inside of the epoch
            performance_measures: performance measures dictionary
            context: model trainer
            validation: should validation dataloader be used for comparison

        """
        if step % self._epoch_steps == 0:  # only record at end of epoch
            return

        for key, value in performance_measures.items():
            if type(value) == torch.Tensor:
                value = value.item()
            self.history[key].append(value)
