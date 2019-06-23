import logging

import torch
import deprecation
import inspect
import torch.nn as nn

from common.class_utils import get_constructor_args, get_class_by_name
from serialization import is_serializable

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


class Loader:

    @deprecation.deprecated(
        details="This type of loading may cause problems when path of model class changes. "
                "Pleas use only when saved with save_model_simple method")
    def load_model_simple(self, path: str):
        """

        Args:
            path:

        Returns:

        """
        return torch.load(path)

    def load_model_from_path(self, cls_path: str, path: str, deserialize_method_name: str = 'deserialize') -> nn.Module:
        """
        Loads the model from pickle file.

        If deserialize_method_name exists the deserialized content of pickle file in path is passed to the
        deserialize_method_name method. In this case,
        the responsibility of creating cls object stays at the caller side.

        Args:
            cls_path (str): name of the class of the model
            path (str): path to pickle-serialized model or model parameters
            deserialize_method_name (str): name of the function that this method should call in order to deserialize the
                model. Must accept single argument of type dict.


        Returns (nn.Module): Loaded model

        """
        return self.load_model(get_class_by_name(cls_path), path, deserialize_method_name)

    def load_model(self, cls: type, path: str, deserialize_method_name: str = 'deserialize') -> nn.Module:
        """
        Loads the model from pickle file.

        If deserialize_method_name exists the deserialized content of pickle file in path is passed to the
        deserialize_method_name method. In this case,
        the responsibility of creating cls object stays at the caller side.

        Args:
            cls (type): class object of the model
            path (str): path to pickle-serialized model or model parameters
            deserialize_method_name (str): name of the function that this method should call in order to deserialize the
                model. Must accept single argument of type dict.


        Returns (nn.Module): Loaded model

        """
        package = torch.load(path, map_location=lambda storage, loc: storage)
        if hasattr(cls, deserialize_method_name) and callable(getattr(cls, deserialize_method_name)):
            return getattr(cls, deserialize_method_name)(package)
        constructor_args = get_constructor_args(cls)
        stored_keys = set(package.keys())
        stored_keys.remove('state_dict')

        args_to_apply = constructor_args & stored_keys
        # If the lengths are not equal it means that there is some inconsistency between save and load
        if len(args_to_apply) != len(constructor_args):
            not_in_constructor = stored_keys - constructor_args
            if not_in_constructor:
                LOGGER.warning(
                    f"Following fields were deserialized "
                    f"but could not be found in constructor of provided class {not_in_constructor}")
            not_in_package = constructor_args - stored_keys
            if not_in_package:
                LOGGER.warning(
                    f"Following fields exist in class constructor "
                    f"but could not be found in serialized package {not_in_package}")

        filtered_package = {key: package[key] for key in stored_keys}
        model = cls(**filtered_package)
        model.load_state_dict(package['state_dict'])
        return model
