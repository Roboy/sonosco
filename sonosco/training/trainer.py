import logging
import torch
import torch.optim.optimizer
import torch.nn.utils.clip_grad as grads

from collections import defaultdict
from typing import Callable, Union, Tuple, List, Any
from torch.utils.data import DataLoader
from .abstract_callback import AbstractCallback
from sonosco.config.global_settings import CUDA_ENABLED
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder


LOGGER = logging.getLogger(__name__)


class ModelTrainer:
    """
    This class handles the training of a pytorch model. It provides convenience
    functionality to add metrics and callbacks and is inspired by the keras API.
    Args:
        model (nn.Module): model to be trained
        optimizer (optim.Optimizer): optimizer used for training, e.g. torch.optim.Adam
        loss (function): loss function that either accepts (model_output, label) or (input, label, model) if custom_model_eval is true
        epochs (int): epochs to train
        train_data_loader (utils.data.DataLoader): training data
        val_data_loader (utils.data.DataLoader, optional): validation data
        custom_model_eval (boolean, optional): enables training mode where the model is evaluated in the loss function
        gpu (int, optional): if not set training runs on cpu, otherwise an int is expected that determines the training gpu
        clip_grads (float, optional): if set training gradients will be clipped at specified norm
    """

    def __init__(self,
                 model: torch.nn.Module,
                 loss: Union[Callable[[Any, Any], Any],
                             Callable[[torch.Tensor, torch.Tensor, torch.nn.Module], float]],
                 epochs: int,
                 train_data_loader: DataLoader,
                 val_data_loader: DataLoader = None,
                 decoder=None,
                 optimizer=torch.optim.Adam,
                 lr: float = 1e-4,
                 custom_model_eval: bool = False,
                 gpu: int = None,
                 clip_grads: float = None,
                 metrics: List[Callable[[torch.Tensor, Any], Union[float, torch.Tensor]]] = None,
                 callbacks: List[AbstractCallback] = None):

        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss = loss
        self._epochs = epochs
        self._metrics = metrics if metrics is not None else list()
        self._callbacks = callbacks if callbacks is not None else list()
        self._gpu = gpu
        self._custom_model_eval = custom_model_eval
        self._clip_grads = clip_grads
        self.decoder = decoder
        self._stop_training = False  # used stop training externally

        if self._gpu is None and CUDA_ENABLED:
            self._gpu = 0

    def set_metrics(self, metrics):
        """
        Set metric functions that receive y_pred and y_true. Metrics are expected to return
        a basic numeric type like float or int.
        """
        self._metrics = metrics

    def add_metric(self, metric):
        self._metrics.append(metric)

    def set_callbacks(self, callbacks):
        """
        Set callbacks that are callable functionals and receive epoch, step, loss, context.
        Context is a pointer to the ModelTrainer instance. Callbacks are called after each
        processed batch.
        """
        self._callbacks = callbacks

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def start_training(self):
        self.model.train()  # train mode
        for epoch in range(1, self._epochs + 1):
            self._epoch_step(epoch)

            if self._stop_training:
                break

        self._close_callbacks()

    def _epoch_step(self, epoch):
        """ Execute one training epoch. """
        running_batch_loss = 0
        running_metrics = defaultdict(float)

        for step, (batch_x, batch_y, input_lengths, target_lengths) in enumerate(self.train_data_loader):
            batch = (batch_x, batch_y, input_lengths, target_lengths)
            batch = self._recursive_to_cuda(batch)  # move to GPU

            # compute training batch
            loss, model_output, grad_norm = self._train_on_batch(batch)
            running_batch_loss += loss.item()

            # compute metrics
            LOGGER.info("Compute Metrics")
            self._compute_running_metrics(model_output, batch, running_metrics)
            running_metrics['gradient_norm'] += grad_norm  # add grad norm to metrics

            # evaluate validation set at end of epoch
            if self.val_data_loader and step == (len(self.train_data_loader) - 1):
                self._compute_validation_error(running_metrics)

            # print current loss and metrics and provide it to callbacks
            performance_measures = self._construct_performance_dict(step, running_batch_loss, running_metrics)
            self._print_step_info(epoch, step, performance_measures)
            self._apply_callbacks(epoch, step, performance_measures)

    def stop_training(self):
        self._stop_training = True

    def _comp_gradients(self):
        """ Compute the gradient norm for all model parameters. """
        grad_sum = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_sum += torch.sum(param.grad ** 2)
        grad_norm = torch.sqrt(grad_sum).item()
        return grad_norm

    def _train_on_batch(self, batch):
        """ Compute loss depending on settings, compute gradients and apply optimization step. """
        # evaluate loss
        batch_x, batch_y, input_lengths, target_lengths = batch
        if self._custom_model_eval:
            loss, model_output = self.loss(batch, self.model)
        else:
            model_output = self.model(batch_x, input_lengths)
            loss = self.loss(model_output, batch_y)

        self.optimizer.zero_grad()  # reset gradients
        loss.backward()  # backpropagation

        # gradient clipping
        if self._clip_grads is not None:
            grads.clip_grad_norm(self.model.parameters(), self._clip_grads)

        grad_norm = self._comp_gradients()  # compute average gradient norm

        self.optimizer.step()  # apply optimization step
        return loss, model_output, grad_norm

    def _compute_validation_error(self, running_metrics):
        """ Evaluate the model's validation error. """
        running_val_loss = 0

        self.model.eval()
        for batch in self.val_data_loader:
            batch = self._recursive_to_cuda(batch)

            # evaluate loss
            batch_x, batch_y = batch
            if self._custom_model_eval:  # e.g. used for sequences and other complex model evaluations
                val_loss, model_output = self.loss(batch, self.model)
            else:
                model_output = self.model(batch_x)
                val_loss = self.loss(model_output, batch_y)

            # compute running validation loss and metrics. add 'val_' prefix to all measures.
            running_val_loss += val_loss.item()
            self._compute_running_metrics(model_output, batch, running_metrics, prefix='val_')
        self.model.train()

        # add loss to metrics and normalize all validation measures
        running_metrics['val_loss'] = running_val_loss
        for key, value in running_metrics.items():
            if 'val_' not in key:
                continue
            running_metrics[key] = value / len(self.val_data_loader)

    def _compute_running_metrics(self,
                                 y_pred: torch.Tensor,
                                 batch: Tuple[torch.Tensor, torch.Tensor],
                                 running_metrics: dict,
                                 prefix: str = ''):
        """
        Computes all metrics based on predictions and batches and adds them to the metrics
        dictionary. Allows to prepend a prefix to the metric names in the dictionary.
        """
        for metric in self._metrics:
            if self._custom_model_eval:
                LOGGER.info(f"Compute metric: {metric.__name__}")
                if metric.__name__ == 'word_error_rate' or metric.__name__ == 'character_error_rate':
                    metric_result = metric(y_pred, batch, self.decoder)
                else:
                    metric_result = metric(y_pred, batch)
            else:
                batch_y = batch[1]
                metric_result = metric(y_pred, batch_y)

            # convert to float if metric returned tensor
            if type(metric_result) == torch.Tensor:
                metric_result = metric_result.item()

            running_metrics[prefix + metric.__name__] += metric_result

    def _construct_performance_dict(self, train_step, running_batch_loss, running_metrics):
        """
        Constructs a combined dictionary of losses and metrics for callbacks based on
        the current running averages.
        """
        performance_dict = defaultdict()
        for key, value in running_metrics.items():
            if 'val_' not in key:
                performance_dict[key] = value / (train_step + 1.)
            else:
                performance_dict[key] = value  # validation metrics, already normalized

        performance_dict['loss'] = running_batch_loss / (train_step + 1.)
        return performance_dict

    def _apply_callbacks(self, epoch, step, performance_measures):
        """ Call all registered callbacks with current batch information. """
        for callback in self._callbacks:
            callback(epoch, step, performance_measures, self)

    def _close_callbacks(self):
        """ Signal callbacks training is finished. """
        for callback in self._callbacks:
            callback.close()

    def _print_step_info(self, epoch, step, performance_measures):
        """ Print running averages for loss and metrics during training. """
        output_message = "epoch {}   batch {}/{}".format(epoch, step, len(self.train_data_loader) - 1)
        delim = "   "
        for metric_name in sorted(list(performance_measures.keys())):
            if metric_name == 'gradient_norm':
                continue
            output_message += delim + "{}: {:.6f}".format(metric_name, performance_measures[metric_name])
        LOGGER.info(output_message)

    def _recursive_to_cuda(self, tensors):
        """
        Recursively iterates nested lists in depth-first order and transfers all tensors
        to specified cuda device.
        Parameters:
            tensors (list or Tensor): list of tensors or tensor tuples, can be nested
        """
        if self._gpu is None:  # keep on cpu
            return tensors

        if type(tensors) != list and type(tensors) != tuple:  # not only for torch.Tensor
            return tensors.to(device=self._gpu)

        cuda_tensors = list()
        for i in range(len(tensors)):
            cuda_tensors.append(self._recursive_to_cuda(tensors[i]))
        return cuda_tensors
