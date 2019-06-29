import logging
import sys
from functools import reduce

import torch
import deprecation
import torch.nn as nn
from common.constants import CLASS_NAME_FIELD, CLASS_MODULE_FIELD

from common.serialization_utils import get_constructor_args, get_class_by_name, is_serialized_primitive, \
    is_serialized_collection, is_serialized_type, raise_unsupported_data_type

LOGGER = logging.getLogger(__name__)


class Loader:

    @deprecation.deprecated(
        details="Use only when model was serialized with serialize_mode_simple")
    def deserialize_model_simple(self, path: str) -> object:
        """
        Loads the model object from the pickle file located at @path.

        Args:
            path (str): path to pickle file containing the model

        Returns (object): deserialized model

        """
        return torch.load(path)

    def deserialize_model_parameters(self, path: str) -> dict:
        """
        Loads the (meta)parameters of the model from the pickle file located at @path

        Args:
            path (str): path to pickle file containing the parameters

        Returns (dict): dictionary of saved parameters

        """
        return torch.load(path, map_location=lambda storage, loc: storage)

    def deserialize_model_from_path(self, cls_path: str, path: str) -> nn.Module:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls_path (str): name of the class of the model
            path (str): path to pickle-serialized model or model parameters

        Returns (nn.Module): Loaded model

        """
        return self.deserialize_model(get_class_by_name(cls_path), path)

    def deserialize_model(self, cls: type, path: str) -> nn.Module:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls (type): class object of the model
            path (str): path to pickle-serialized model or model parameters

        Returns (nn.Module): Loaded model

        """
        package = self.deserialize_model_parameters(path)
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
