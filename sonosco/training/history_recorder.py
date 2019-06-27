import logging
import torch

from collections import defaultdict
from .abstract_callback import AbstractCallback


LOGGER = logging.getLogger(__name__)


class HistoryRecorder(AbstractCallback):
    """ Records all losses and metrics during training. """

    def __init__(self, epoch_steps):
        self.history = defaultdict(list)
        self._epoch_steps = epoch_steps

    def __call__(self, epoch, step, performance_measures, context):

        if step % self._epoch_steps == 0:  # only record at end of epoch
            return

        for key, value in performance_measures.items():
            if type(value) == torch.Tensor:
                value = value.item()
            self.history[key].append(value)
