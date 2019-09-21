import logging

from typing import Dict
from sonosco.serialization import serializable
from ..abstract_callback import AbstractCallback, ModelTrainer

LOGGER = logging.getLogger(__name__)


@serializable
class DisableSoftWindowAttention(AbstractCallback):
    """
    Disable soft window pretraining after the stecified disable epoch.

    Args:
        disable_epoch: epoch after which soft-window pretraining should be disabled

    """
    disable_epoch: int = 3

    def __call__(self,
                 epoch: int,
                 step: int,
                 performance_measures: Dict,
                 context: ModelTrainer,
                 validation: bool = False) -> None:
        """
        Disable soft window if epoch greater than disable epoch.

        Args:
            epoch: epoch step
            step: step inside of the epoch
            performance_measures: performance measures dictionary
            context: model trainer
            validation: should validation dataloader be used for comparison

        """

        if epoch > self.disable_epoch:
            context.model.decoder.soft_window_enabled = False
