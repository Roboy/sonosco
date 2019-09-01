import logging
import torch
import deprecation
import torch.nn as nn

from sonosco.common.constants import SONOSCO_CONFIG_SERIALIZE_NAME
from sonosco.model.serialization import is_serializable, serializable

LOGGER = logging.getLogger(__name__)


@serializable
class Serializer:

    def __post_init__(self) -> None:
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

    def serialize(self, obj: object, path: str, config: dict = None) -> None:
        """
        Saves the model using pickle protocol.

        It requires the model to have @sonosco.serialization.serializable annotation at the class definition level.

        Saves dictionary with all the (meta)parameters of the model.

        Args:
            obj (nn.Module): model to save
            path (str) : path where to save the model
            config (dict): optional configuration for the object
        Returns:

        """
        if is_serializable(obj):
            entity_to_save = obj.__serialize__()
            if config:
                torch.save({obj.__class__.__name__: entity_to_save, SONOSCO_CONFIG_SERIALIZE_NAME: config}, path)
            else:
                torch.save(entity_to_save, path)
        else:
            raise TypeError("Only @serializable class can be serialized")
