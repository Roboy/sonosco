import logging
from sonosco.training.abstract_callback import  AbstractCallback
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class TensorBoardCallback(AbstractCallback):

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):

        for key, value in performance_measures.items():
            LOGGER.info(f"performance measure: {key}, {value}")
            self.writer.add_scalar(key, value)
