import logging

from typing import Dict
from sonosco.serialization import serializable
from sonosco.training.abstract_callback import AbstractCallback, ModelTrainer
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class TensorBoardCallback(AbstractCallback):
    """
    Plot all metrics in tensorboard.

    Args:
        log_dir: logging directory for tensorboard
    """
    log_dir: str

    def __post_init__(self) -> None:
        """
        Post initialization.
        """
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __call__(self,
                 epoch: int,
                 step: int,
                 performance_measures: Dict,
                 context: ModelTrainer,
                 validation: bool = False) -> None:
        """
        Perform plotting.

        Args:
            epoch: epoch step
            step: step inside of the epoch
            performance_measures: performance measures dictionary
            context: model trainer
            validation: should validation dataloader be used for comparison

        """
        if step % context.test_step > 0:
            return

        for key, value in performance_measures.items():
            LOGGER.info(f"performance measure: {key}, {value}")
            self.writer.add_scalar(key, value)
