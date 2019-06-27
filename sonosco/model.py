import logging
import sys
from functools import reduce

import torch
import deprecation
import torch.nn as nn
from common.constants import CLASS_NAME_FIELD, CLASS_MODULE_FIELD

from common.serialization_utils import get_constructor_args, get_class_by_name, is_serialized_primitive, \
    is_serialized_collection, is_serialized_type, raise_unsupported_data_type
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

    def load_model_parameters(self, path: str) -> dict:
        """

        Args:
            path:

        Returns (dict): dictionary of saved parameters

        """
        return torch.load(path, map_location=lambda storage, loc: storage)

    def load_model_from_path(self, cls_path: str, path: str) -> nn.Module:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls_path (str): name of the class of the model
            path (str): path to pickle-serialized model or model parameters

        Returns (nn.Module): Loaded model

        """
        return self.load_model(get_class_by_name(cls_path), path)

    def load_model(self, cls: type, path: str) -> nn.Module:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls (type): class object of the model
            path (str): path to pickle-serialized model or model parameters

        Returns (nn.Module): Loaded model

        """
        package = self.load_model_parameters(path)
        constructor_args_names = get_constructor_args(cls)
        serialized_args_names = set(package.keys())
        serialized_args_names.discard('state_dict')

        args_to_apply = constructor_args_names & serialized_args_names

        not_in_constructor = serialized_args_names - constructor_args_names
        if not_in_constructor:
            LOGGER.warning(
                f"Following fields were deserialized "
                f"but could not be found in constructor of provided class {not_in_constructor}")
        not_in_package = constructor_args_names - serialized_args_names
        if not_in_package:
            LOGGER.warning(
                f"Following fields exist in class constructor "
                f"but could not be found in serialized package {not_in_package}")

        kwargs = {}

        for arg in args_to_apply:
            serialized_val = package.get(arg)
            # TODO: Rewrite this ugly if else chain to something more OO
            if is_serialized_type(serialized_val):
                kwargs[arg] = reduce(getattr,
                                     f"{serialized_val[CLASS_MODULE_FIELD]}.{serialized_val[CLASS_NAME_FIELD]}"
                                     .split("."), sys.modules[__name__])
            elif is_serialized_primitive(serialized_val) or is_serialized_collection(serialized_val):
                kwargs[arg] = serialized_val
            else:
                raise_unsupported_data_type()

        model = cls(**kwargs)
        model.load_state_dict(package['state_dict'])
        return model
