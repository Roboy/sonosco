import logging

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


@serializable
class TensorBoardCallback(AbstractCallback):
    log_dir: str

    def __post_init__(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):
        if step % context.test_step > 0:
            return

        for key, value in performance_measures.items():
            LOGGER.info(f"performance measure: {key}, {value}")
            self.writer.add_scalar(key, value)
