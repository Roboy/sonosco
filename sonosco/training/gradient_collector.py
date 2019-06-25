import logging
import numpy as np
import torch

from .abstract_callback import AbstractCallback


LOGGER = logging.getLogger(__name__)


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
