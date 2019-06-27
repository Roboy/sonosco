import logging
import torch
import deprecation
import torch.nn as nn

from sonosco.models.serialization import is_serializable

LOGGER = logging.getLogger(__name__)


class Saver:

    def __init__(self) -> None:
        super().__init__()

    @deprecation.deprecated(
        details="This type of saving may cause problems when path of model class changes. Pleas use save_model instead")
    def save_model_simple(self, model: nn.Module, path: str) -> None:
        """
       Simply saves the model using pickle protocol.
        Args:
            model (nn.Module): model to save
            path (str) : path where to save the model

        Returns:

        """
        torch.save(model, path)

    def save_model(self, model: nn.Module, path: str) -> None:
        """
        Saves the model using pickle protocol.

        It requires the model to have @sonosco.serialization.serializable annotation at the class definition level.

        Args:
            model (nn.Module): model to save
            path (str) : path where to save the model
        Returns:

        """
        if is_serializable(model):
            entity_to_save = model.__serialize__()
            torch.save(entity_to_save, path)
        else:
            raise TypeError("Only @serializable class can be serialized")

