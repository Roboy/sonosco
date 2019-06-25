import logging
import torch
import deprecation
import torch.nn as nn

from .serialization import is_serializable

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

        If the infer_structure is True this method infers all the meta parameters of the model and save them together
        with learnable parameters.

        If the infer_structure is False and method specified by serialize_method_name exists, the return value of the
        serialize_method_name method is saved.

        If neither of above only learnable parameters a.k.a. state_dict are saved.

        Args:
            model (nn.Module): model to save
            path (str) : path where to save the model
            infer_structure (bool): indicator whether to infer the model structure
            serialize_method_name (str): name of the function that this method should call in order to serialize the
                model. Must return dict.

        Returns:

        """
        if is_serializable(model):
            entity_to_save = model.__serialize__()
            torch.save(entity_to_save, path)
        else:
            raise TypeError("Only @serializable class can be serialized")
