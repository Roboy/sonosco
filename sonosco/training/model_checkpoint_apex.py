import logging
import sys
import os.path as path
import torch

from apex import amp
from .abstract_callback import AbstractCallback
from sonosco.model.serializer import ModelSerializer


LOGGER = logging.getLogger(__name__)


class ModelCheckpointApex(AbstractCallback):
    """
    Saves the model and optimizer state at the point with lowest validation error throughout training.
    Args:
        output_path (string): path to directory where the checkpoint will be saved to
        model_name (string): name of the checkpoint file
    """

    def __init__(self, output_path, model_name='model_checkpoint.pt'):
        self.output_path = output_path
        self.model_name = model_name
        self.best_val_score = sys.float_info.max
        self.serializer = ModelSerializer()

    def __call__(self, epoch, step, performance_measures, context):
        if step == (len(context.train_data_loader) - 1):
            LOGGER.info(f"Saving model checkpoint after epoch {epoch}.")
            self._save_checkpoint(context, path.join(self.output_path, f"model_epoch_{epoch}.pt"))

        if 'val_loss' not in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_checkpoint(context, path.join(self.output_path, self.model_name))

    def _save_checkpoint(self, context, output_path):
        LOGGER.info("Saving model at checkpoint.")
        context.model.eval()
        checkpoint = {
            'model': context.model.state_dict(),
            'optimizer': context.optimizer.state_dict(),
            'amp': amp.state_dict()
        }
        torch.save(checkpoint, output_path)
        context.model.train()
