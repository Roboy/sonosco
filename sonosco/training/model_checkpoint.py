import logging
import sys
import os.path as path
from typing import Dict

from sonosco.model.serialization import serializable

from .abstract_callback import AbstractCallback
from sonosco.model.serializer import Serializer

LOGGER = logging.getLogger(__name__)


@serializable
class ModelCheckpoint(AbstractCallback):
    """
    Saves the model and optimizer state at the point with lowest validation error throughout training.
    Args:
        output_path (string): path to directory where the checkpoint will be saved to
        model_name (string): name of the checkpoint file
    """
    output_path: str
    config: Dict[str, object] = None
    model_name: str = 'model_checkpoint.pt'
    trainer_name: str = 'trainer_checkpoint.pt'

    def __post_init__(self):
        self.best_val_score = sys.float_info.max
        self.serializer = Serializer()

    def __call__(self, epoch, step, performance_measures, context):
        if step == (len(context.train_data_loader) - 1):
            LOGGER.info(f"Saving model checkpoint after epoch {epoch}.")
            self._save_checkpoint(context.model, path.join(self.output_path, f"model_epoch_{epoch}.pt"))
            self._save_checkpoint(context, path.join(self.output_path, f"trainer_epoch_{epoch}.pt"))

        if 'val_loss' not in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_checkpoint(context.model, path.join(self.output_path, self.model_name))
            self._save_checkpoint(context, path.join(self.output_path, self.trainer_name))

    def _save_checkpoint(self, obj, output_path):
        LOGGER.info("Saving model at checkpoint.")
        if self.config:
            self.serializer.serialize(obj=obj, path=output_path, config=self.config)
        else:
            self.serializer.serialize(obj=obj, path=output_path)
