import logging
import torch
import torch.nn
import torch.optim.optimizer
import math
import os
import json
import numpy as np

from dataclasses import field, dataclass
from torch.utils.data import RandomSampler
from collections import defaultdict
from sonosco.common.constants import SONOSCO


from typing import Callable, Union, Tuple, List, Any

from sonosco.decoders.decoder import Decoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)

@dataclass
class ModelEvaluator:
    """
    This class handles the the evaluation of a pytorch model. It provides convenience
    functionality to add metrics and calculates mean and variance of these metrics
    using the bootstrapping method.
    Args:
        model (nn.Module): model to be evaluated
        data_loader (utils.data.DataLoader): test data, needs to have a random sampler as batch_sampler or smapler
        bootstrap_size (int): number of samples that are contained in one bo
        num_bootstraps (int): number of boostraps to compute
        decoder (sonosco.decoders.decoder, optional): decoder to decode model output in order to calculate wer and cer
        metrics (list of metrics, optional): metrics that are supposed to be evaluated (need to be functions that get model_output, batch (and decoder)
    """
    model: torch.nn.Module
    data_loader: DataLoader
    bootstrap_size: int
    num_bootstraps: int
    decoder: Decoder = None
    device: torch.device = None
    metrics: List[Callable[[torch.Tensor, Any], Union[float, torch.Tensor]]] = field(default_factory=list)
    _current_bootstrap_step: int = None
    _eval_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        self._check_replacement_sampler_in_dataloader()
        self._evaluation_done = False

    def _check_replacement_sampler_in_dataloader(self):
        '''
        check if provided dataloader contains a randomsampler
        '''
        if not (isinstance(self.data_loader.sampler,RandomSampler) or isinstance(self.data_loader.batch_sampler,RandomSampler)):
            LOGGER.info(f'No random sampler in dataloader.')
            assert()

    def _bootstrap_step(self, mean_dict):
        '''
        computes metrics for all steps in one bootstrap step
        '''
        torch.no_grad()
        running_metrics = {metric.__name__: [] for metric in self.metrics}

        for sample_step in range(self.bootstrap_size):
            batch_x, batch_y, input_lengths, target_lengths = next(iter(self.data_loader))

            batch = (batch_x, batch_y, input_lengths, target_lengths)
            batch = self._recursive_to_cuda(batch)  # move to GPU
            batch_x, batch_y, input_lengths, target_lengths = batch

            model_output = self.model(batch_x, input_lengths)

            self._compute_running_metrics(model_output, batch, running_metrics)

        self._fill_mean_dict(running_metrics, mean_dict)

    def _compute_running_metrics(self,
                                 model_output: torch.Tensor,
                                 batch: Tuple[torch.Tensor, torch.Tensor],
                                 running_metrics: dict):
        """
        Computes all metrics based on predictions and batches and adds them to the metrics
        dictionary. Allows to prepend a prefix to the metric names in the dictionary.
        """
        for metric in self.metrics:
            if metric.__name__ == 'word_error_rate' or metric.__name__ == 'character_error_rate':
                metric_result = metric(model_output, batch, self.decoder)
            else:
                metric_result = metric(model_output, batch)
            if type(metric_result) == torch.Tensor:
                metric_result = metric_result.item()

            running_metrics[metric.__name__].append(metric_result)

    def _fill_mean_dict(self, running_metrics, mean_dict):
        '''
        calculates the mean and saves it in a dictionary
        '''
        for key, value in running_metrics.items():
            mean = np.mean(value)
            mean_dict[key].append(mean)

    def _compute_mean_variance(self, mean_dict):
        '''
        compute mean and variance of each list in the dictionary
        '''
        self.eval_dict = defaultdict()
        for key, value in mean_dict.items():
            tmp_mean = np.mean(value)
            tmp_variance =np.var(value)
            self.eval_dict[key + '_mean'] = tmp_mean
            self.eval_dict[key + '_variance'] = tmp_variance

    def set_metrics(self, metrics):
        """
        Set metric functions that receive y_pred and y_true. These metrics are used to
        create a statistical evaluation of the model provided
        """
        self.metrics = metrics

    def add_metric(self, metric):
        self.metrics.append(metric)

    def start_evaluation(self, tb_path: str = None, log_path: str = None):
        '''
        start model evaluation for all metrics
        :param tb_path: (str, oiptional) - path to log directory to store tb logs , if provided, tensorboard logs are written
        :param log_path: (str, optional) - path to log directory, if provided, evaluation results will be dumped in json format
        '''
        LOGGER.info(f'Start Evaluation')
        self.model.eval() #evaluation mode
        mean_dict = {metric.__name__: [] for metric in self.metrics}

        for bootstrap_step in range(self.num_bootstraps):
            self._current_bootstrap_step = bootstrap_step
            self._bootstrap_step(mean_dict)
        self._compute_mean_variance(mean_dict)
        self._evaluation_done = True
        if tb_path is not None:
            self.dump_to_tensorboard(tb_path)
        if log_path is not None:
            self.dump_evaluation(log_path)

    def dump_evaluation(self, output_path):
        '''
        dump evaluation dict to output path
        '''
        if self._evaluation_done == False:
            self.start_evaluation()
        file_to_dump = os.path.join(output_path, 'evaluation.json')
        LOGGER.info(f'dump evaluation results to {file_to_dump}')
        with open(file_to_dump, 'w') as fp:
            json.dump(self.eval_dict, fp)

    def dump_to_tensorboard(self, log_path):
        '''
        write tensorboard logs for evaluation results to log_path
        '''
        LOGGER.info(f'Log evaluations in tensorboard.')
        writer = SummaryWriter(log_dir=log_path)
        for key, value in self.eval_dict:
            writer.add_scalar(key, value)

    def _recursive_to_cuda(self, tensors):
        """
        Recursively iterates nested lists in depth-first order and transfers all tensors
        to specified cuda device.
        Parameters:
            tensors (list or Tensor): list of tensors or tensor tuples, can be nested
        """
        if self.device is None:  # keep on cpu
            return tensors

        if type(tensors) != list and type(tensors) != tuple:  # not only for torch.Tensor
            return tensors.to(device=self.device)

        cuda_tensors = list()
        for i in range(len(tensors)):
            cuda_tensors.append(self._recursive_to_cuda(tensors[i]))
        return cuda_tensors





