import logging
import sys
import os.path as path
import numpy as np
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


class LayerwiseGradientNorm(AbstractCallback):
    """ Collects the layer-wise gradient norms for each epoch. """

    def __init__(self):
        self.layer_grads = dict()
        self._batch_layer_grads = dict()

    def __call__(self, epoch, step, performance_measures, context):
        """
        Store gradient norms for each batch and compute means after the
        epoch's last batch.
        """
        self._store_batch_layer_grads(context.model)

        if step == (len(context.train_data_loader) - 1):    # end of epoch
            self._store_layer_grads()
            self._batch_layer_grads = dict()

    def _store_batch_layer_grads(self, model):
        """ Store gradient norm of each layer for current batch. """
        for name, param in model.named_parameters():

            if not param.requires_grad or param.grad is None:
                continue

            if not name in self._batch_layer_grads:
                self._batch_layer_grads[name] = []

            grad_norm = torch.sqrt(torch.sum(param.grad**2)).item()
            self._batch_layer_grads[name].append(grad_norm)

    def _store_layer_grads(self):
        """ Compute mean of all batch steps in epoch. """
        for name, grads in self._batch_layer_grads.items():

            if name not in self.layer_grads:
                self.layer_grads[name] = []

            layer_epoch_grad = np.mean(grads)
            self.layer_grads[name].append(layer_epoch_grad)


class EarlyStopping(AbstractCallback):
    """
    Early Stopping to terminate training early if the monitored metric did not improve
    over a number of epochs.
    Args:
        monitor (string): name of the relevant loss or metric (usually 'val_loss')
        min_delta (float): minimum change in monitored metric to qualify as an improvement
        patience (int): number of epochs to wait for an improvement before terminating the training
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.last_best = sys.float_info.max
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, epoch, step, performance_measures, context):

        if step != len(context.train_data_loader) - 1:  # only continue at end of epoch
            return

        if self.monitor not in performance_measures:
            return

        current_loss = performance_measures[self.monitor]
        if (self.last_best - current_loss) >= self.min_delta:
            self.last_best = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            context._stop_training = True   # make ModelTrainer stop
            LOGGER.info(f"Early stopping after epoch {epoch}")
