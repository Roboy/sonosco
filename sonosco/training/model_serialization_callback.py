import logging
import sys
import os.path as path
import torch

from .abstract_callback import AbstractCallback
from sonosco.model.serializer import ModelSerializer
from sonosco.model.serialization import serializable


LOGGER = logging.getLogger(__name__)

class ModelSerializationCallback(AbstractCallback):
    """
    Saves the model and optimizer state at the point with lowest validation error throughout training.
    Args:
        output_path (string): path to directory where the checkpoint will be saved to
        model_name (string): name of the checkpoint file
    """
    def __init__(self, output_path, model_name='model_checkpoint'):
        self.output_path = path.join(output_path, model_name)
        self.best_val_score = sys.float_info.max
        self.serializer = ModelSerializer()

    def __call__(self, epoch, step, performance_measures, context):
        if 'val_loss' not in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_serialized_model(context.model)

    def _save_serialized_model(self, model):
        LOGGER.info("Saving model at checkpoint.")
        model.eval()
        self.serializer.serialize_model(model=model, path=self.output_path)
