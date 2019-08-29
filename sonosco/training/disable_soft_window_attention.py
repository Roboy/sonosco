import logging

from sonosco.model.serialization import serializable

from sonosco.training.abstract_callback import AbstractCallback

LOGGER = logging.getLogger(__name__)


@serializable
class DisableSoftWindowAttention(AbstractCallback):
    disable_epoch: int = 3

    def __call__(self,
                 epoch,
                 step,
                 performance_measures,
                 context,
                 validation: bool = False):

        if epoch > self.disable_epoch:
            context.model.decoder.soft_window_enabled = False
