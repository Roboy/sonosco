from typing import TypeVar
from abc import ABC, abstractmethod
from sonosco.serialization import serializable


ModelTrainer = TypeVar("ModelTrainer")


@serializable
class AbstractCallback(ABC):
    """
    Interface that defines how callbacks must be specified.
    """

    @abstractmethod
    def __call__(self, epoch: int, step: int, performance_measures: dict, context: ModelTrainer) -> None:
        """
        Called after every batch by the ModelTrainer.

        Args:
            epoch (int): current epoch number
            step (int): current batch number
            performance_measures (dict): losses and metrics based on a running average
            context (ModelTrainer): reference to the calling ModelTrainer, allows to access members

        """
        pass

    def close(self) -> None:
        """
        Handle cleanup work if necessary. Will be called at the end of the last epoch.
        """
        pass
