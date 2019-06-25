import logging
import sys
import os.path as path
import torch

from .abstract_callback import AbstractCallback


LOGGER = logging.getLogger(__name__)


class ModelCheckpoint(AbstractCallback):
    """
    Saves the model and optimizer state at the point with lowest validation error throughout training.
    Args:
        output_path (string): path to directory where the checkpoint will be saved to
        model_name (string): name of the checkpoint file
    """

    def __init__(self, output_path, model_name='model_checkpoint.pt'):
        self.output_path = path.join(output_path, model_name)
        self.best_val_score = sys.float_info.max

    def __call__(self, epoch, step, performance_measures, context):

        if 'val_loss' not in performance_measures:
            return

        if performance_measures['val_loss'] < self.best_val_score:
            self.best_val_score = performance_measures['val_loss']
            self._save_checkpoint(context.model, context.optimizer, epoch)

    def _save_checkpoint(self, model, optimizer, epoch):
        LOGGER.info("Saving model at checkpoint.")
        model.eval()
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        torch.save({'arch': model.__class__.__name__,
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict
                    }, self.output_path)
        model.train()
