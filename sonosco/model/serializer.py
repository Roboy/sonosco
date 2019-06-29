import logging
import torch
import deprecation
import torch.nn as nn

from sonosco.model.serialization import is_serializable

LOGGER = logging.getLogger(__name__)


class ModelSerializer:

    def __init__(self) -> None:
        super().__init__()

    @deprecation.deprecated(
        details="This way of saving may cause some problems when the path of the model class changes. "
                "Pleas use save_model if possible")
    def serialize_model_simple(self, model: nn.Module, path: str) -> None:
        """
        Simply saves the model using pickle protocol.
        By using this method whole state of the object is saved together with reference (path) to the class definition.
        When using this method deserialization is only possible to *exactly* the same class.
        This method provides little flexibility and may cause problems when distributing the model.


        Args:
            model (nn.Module): model to save
            path (str) : path where to save the model

        Returns:

        """
        torch.save(model, path)

    def serialize_model(self, model: nn.Module, path: str) -> None:
        """
        Saves the model using pickle protocol.

        It requires the model to have @sonosco.serialization.serializable annotation at the class definition level.

        Saves dictionary with all the (meta)parameters of the model.

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
