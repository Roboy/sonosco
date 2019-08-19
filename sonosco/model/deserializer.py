import inspect
import logging
import sys
import torch
import deprecation
import torch.nn as nn

from typing import Dict, Any
from functools import reduce
from sonosco.common.constants import CLASS_NAME_FIELD, CLASS_MODULE_FIELD, SERIALIZED_FIELD
from sonosco.common.serialization_utils import get_constructor_args, get_class_by_name, is_serialized_primitive, \
    is_serialized_collection, is_serialized_type, raise_unsupported_data_type, is_serialized_dataclass

LOGGER = logging.getLogger(__name__)


class ModelDeserializer:

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
        caller_module = inspect.getmodule(inspect.stack()[1][0])
        return ModelDeserializer.__deserialize_model(cls, package, caller_module)

    @staticmethod
    def __deserialize_model(cls: type, package: Dict[str, Any], caller_module: object) -> nn.Module:
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
            if is_serialized_dataclass(serialized_val):
                clazz = ModelDeserializer.__create_class_object(
                    f"{serialized_val[CLASS_MODULE_FIELD]}.{serialized_val[CLASS_NAME_FIELD]}", caller_module)
                kwargs[arg] = ModelDeserializer.__deserialize_model(clazz, serialized_val[SERIALIZED_FIELD],
                                                                    caller_module)

            elif is_serialized_type(serialized_val):
                kwargs[arg] = ModelDeserializer.__create_class_object(
                    f"{serialized_val[CLASS_MODULE_FIELD]}.{serialized_val[CLASS_NAME_FIELD]}", caller_module)

            elif is_serialized_primitive(serialized_val) or is_serialized_collection(serialized_val):
                kwargs[arg] = serialized_val

            else:
                raise_unsupported_data_type()

        obj = cls(**kwargs)
        if package.get('state_dict'):
            obj.load_state_dict(package['state_dict'])
        return obj

    @staticmethod
    def __create_class_object(full_class_name: str, caller_module: object):
        # todo: Add import of the package of the class to serialize (if necessary)
        # Creates class object (type) from full_class_name
        # In current module we have only access to top level module (e.g torch)
        # but we want to create particular class.
        # The reduce method with call getattr with more nested path on each iteration
        # (e.g torch, torch.nn, torch.nn.modules, etc.).

        # todo: More generic approach should be implemented. For now only caller's scope and current scope are checked.
        try:
            return reduce(getattr, full_class_name.split("."), sys.modules[__name__])
        except Exception as e:
            LOGGER.info("could not find appropriate class in current module")

        return reduce(getattr, full_class_name.split(".")[1:], caller_module)

    @staticmethod
    def __reduce_from_module(full_class_name: str, module: object):
        reduce(getattr, full_class_name.split("."), module)
