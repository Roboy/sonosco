import os
import os.path as path
import datetime
import torch
import logging
import numpy as np

import sonosco.common.path_utils as path_utils
import sonosco.common.utils as utils

from random import random
from typing import Callable, Union, Tuple, List, Any
from sonosco.training import ModelTrainer
from torch.utils.data import DataLoader
from sonosco.training.model_checkpoint import ModelCheckpoint
from sonosco.training.tensorboard_callback import TensorBoardCallback
from .abstract_callback import AbstractCallback

from time import time


LOGGER = logging.getLogger(__name__)


class Experiment:
    """
    Generates a folder where all experiments will be stored an then a named experiment with current
    timestamp and provided name. Automatically starts logging the console output and creates a copy
    of the currently executed code in the experiment folder. The experiment's subfolder paths are provided
    to the outside as member variables. It also allows adding of more subfolders conveniently.
    Args:
        experiment_name (string): name of the exerpiment to be created
        experiments_path (string): location where all experiments will be stored, default is './experiments'
    Example:
        >>> experiment = Experiment('mnist_classification')
        >>> print(experiment.plots) # path to experiment plots
    """

    def __init__(self,
                 experiment_name,
                 seed:int = None,
                 experiments_path=None,
                 sub_directories=("plots", "logs", "code", "checkpoints"),
                 exclude_dirs=('__pycache__', '.git', 'experiments'),
                 exclude_files=('.pyc',)):

        self.experiments_path = self._set_experiments_dir(experiments_path)
        self.name = self._set_experiment_name(experiment_name)
        if seed is not None:
            self._set_seed(seed)
        self.path = path.join(self.experiments_path, self.name)     # path to current experiment
        self.logs = path.join(self.experiments_path, "logs")
        self.plots_path = path.join(self.experiments_path, "plots")
        self.checkpoints_path = path.join(self.experiments_path, "checkpoints")

        self.code = path.join(self.experiments_path, "code")
        self._sub_directories = sub_directories

        self._exclude_dirs = exclude_dirs
        self._exclude_files = exclude_files

        self._init_directories()
        self._copy_sourcecode()
        self._set_logging()

    @staticmethod
    def _set_experiments_dir(experiments_path):
        if experiments_path is not None:
            return experiments_path

        local_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        local_path = local_path if local_path != '' else './'
        return path.join(local_path, "experiments")

    @staticmethod
    def _set_experiment_name(experiment_name):
        date_time = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
        return f"{date_time}_{experiment_name}"

    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _set_logging(self):
        utils.add_log_file(path.join(self.logs, "logs"), LOGGER)

    def _init_directories(self):
        """ Create all basic directories. """
        path_utils.try_create_directory(self.experiments_path)
        path_utils.try_create_directory(path.join(self.experiments_path, self.name))
        for sub_dir_name in self._sub_directories:
            self.add_directory(sub_dir_name)

    def _add_member(self, key, value):
        """ Add a member variable named 'key' with value 'value' to the experiment instance. """
        self.__dict__[key] = value

    def _copy_sourcecode(self):
        """ Copy code from execution directory in experiment code directory. """
        sources_path = os.path.dirname(os.path.dirname(__file__))
        sources_path = sources_path if sources_path != '' else './'
        utils.copy_code(sources_path, self.code,
                        exclude_dirs=self._exclude_dirs,
                        exclude_files=self._exclude_files)

    def add_directory(self, dir_name):
        """
        Add a sub-directory to the experiment. The directory will be automatically
        created and provided to the outside as a member variable.
        """
        # store in sub-dir list
        if dir_name not in self._sub_directories:
            self._sub_directories.append(dir_name)
        # add as member
        dir_path = path.join(self.experiments_path, self.name, dir_name)
        self._add_member(dir_name, dir_path)
        # create directory
        path_utils.try_create_directory(dir_path)

    def setup_model_trainer(self,
                            name: str,
                            model: torch.nn.Module,
                            loss: Union[Callable[[Any, Any], Any],
                                        Callable[[torch.Tensor, torch.Tensor, torch.nn.Module], float]],
                            epochs: int,
                            train_data_loader: DataLoader,
                            model_checkpoints: bool = True,
                            tensorboard: bool = True,
                            val_data_loader: DataLoader = None,
                            decoder  = None,
                            optimizer=torch.optim.Adam,
                            lr: float = 1e-4,
                            custom_model_eval: bool = False,
                            gpu: int = None,
                            clip_grads: float = None,
                            metrics: List[Callable[[torch.Tensor, Any], Union[float, torch.Tensor]]] = None,
                            callbacks: List[AbstractCallback] = None):
        """
        Setup a model_trainer object with specified parameters, by default with checkpoint
        callback and tensorboard callback. Add this model trainer to the modeltrainers dictionary.
        """

        trainer = ModelTrainer(model=model, loss=loss, epochs=epochs,
                               train_data_loader=train_data_loader,
                               val_data_loader=val_data_loader, decoder=decoder,
                               optimizer=optimizer, lr=lr, custom_model_eval=custom_model_eval,
                               gpu=gpu, clip_grads=clip_grads,
                               metrics=metrics, callbacks=callbacks)
        if model_checkpoints:
            date_time = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
            trainer.add_callback(ModelCheckpoint(output_path=self.checkpoints))

        if tensorboard:
            trainer.add_callback(TensorBoardCallback(log_dir=self.logs))

        self.model_trainers = {name: trainer}

    def start_training(self, trainer_name: str = None):
        """
        Starts training of all model trainers. If a trainer_name is specified, just this specific
        model trainer is started.

        :param trainer_name: str - if specified just the model trainer with this name gets started
        """
        if trainer_name is None:
            for trainer in self.model_trainers.values():
                trainer.start_training()
        else:
            self.model_trainers[trainer_name].start_training()
        #TODO: add serialization after training is finished

    @staticmethod
    def add_file(folder_path, filename, content):
        """ Adds a file with provided content to folder. Convenience function. """
        with open(path.join(folder_path, filename), 'w') as text_file:
            text_file.write(content)

    @staticmethod
    def create(config: dict):
        """
        :param config: dict - specify a .yaml config with one or more parameters: name, seed,
        experiment_path, sub_dirs, exclude_dirs, exclude_files and read it in as a dictionary.
        :return: Experiment with configuration specified in config dictionary
        """
        date_time = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
        name = config.get('name', default='experiment')
        seed = config.get('global_seed', default=None)
        experiment_path = config.get('experiment_path', default=None)
        sub_dirs = config.get('sub_dirs', default=("plots", "logs", "code", "checkpoints"))
        exclude_dirs = config.get('exclude_dirs', default=(('__pycache__', '.git', 'experiments'),))
        exclude_files = config.get('exclude_files', default=('.pyc',))

        return Experiment(name, seed, experiment_path, sub_dirs, exclude_dirs, exclude_files)
