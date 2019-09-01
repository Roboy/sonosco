import inspect
import logging
import sys
import torch
import deprecation
from typing import Dict, Any
from functools import reduce

from sonosco.common.constants import SONOSCO_CONFIG_SERIALIZE_NAME
from sonosco.common.constants import CLASS_NAME_FIELD, CLASS_MODULE_FIELD, SERIALIZED_FIELD
from sonosco.common.serialization_utils import get_constructor_args, get_class_by_name, is_serialized_primitive, \
    is_serialized_collection, is_serialized_type, raise_unsupported_data_type, is_serialized_dataclass, \
    is_serialized_collection_of_serializables, is_serialized_collection_of_callables

# TODO: improve this
import sonosco  # don't remove, it's for object creation

LOGGER = logging.getLogger(__name__)


# TODO: Extend the scope of callables deserialization, so it can account for changes in module path
class Deserializer:

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

    def deserialize_from_class_path(self, cls_path: str,
                                    path: str,
                                    external_args: Dict[str, object] = None) -> object:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls_path (str): name of the class of the model
            path (str): path to pickle-serialized model or model parameters
            external_args (dict): additional arguments to override serialized or provide custom ones

        Returns (nn.Module): Loaded model

        """
        return self.deserialize(get_class_by_name(cls_path), path, external_args)

    def deserialize(self, cls: type,
                    path: str,
                    external_args: Dict[str, object] = None,
                    with_config: bool = False) -> object:
        """
        Loads the model from pickle file.

        It requires serialization format by @sonosco.serialization.serializable annotation.

        Args:
            cls (type): class object of the model
            path (str): path to pickle-serialized model or model parameters
            external_args (dict): additional arguments to override serialized or provide custom ones

        Returns (nn.Module): Loaded model

        """
        package = self.deserialize_model_parameters(path)
        caller_module = inspect.getmodule(inspect.stack()[1][0])
        return Deserializer.__deserialize(cls, package, caller_module, external_args, with_config)

    @staticmethod
    def __deserialize(cls: type, package: Dict[str, Any],
                      caller_module: object,
                      external_args: Dict[str, object] = None,
                      with_config: bool = False) -> object:

        if external_args is None:
            external_args = {}

        if SONOSCO_CONFIG_SERIALIZE_NAME in package.keys():
            config = package.pop(SONOSCO_CONFIG_SERIALIZE_NAME)
            package = package.popitem()[1]
        else:
            config = None

        if with_config and config is None:
            LOGGER.warning("with_config is True but there was no config serialized under given path")

        constructor_args_names = get_constructor_args(cls)
        serialized_args_names = set(package.keys())
        serialized_args_names.discard('state_dict')

        args_to_apply = constructor_args_names & serialized_args_names
        extra_args_names = set(external_args.keys())

        not_in_constructor = (serialized_args_names | extra_args_names) - constructor_args_names
        if not_in_constructor:
            LOGGER.warning(
                f"Following fields were found "
                f"but could not be found in constructor of provided class {not_in_constructor}")
        not_in_package = constructor_args_names - (serialized_args_names | extra_args_names)
        if not_in_package:
            LOGGER.warning(
                f"Following fields exist in class constructor "
                f"but could not be found in provided arguments {not_in_package}")

        args_to_apply = args_to_apply - extra_args_names

        kwargs = {}

        for arg in args_to_apply:
            serialized_val = package.get(arg)

            # TODO: Rewrite this ugly if else chain to something more OO
            # TODO: This now also catches callable classes
            if is_serialized_dataclass(serialized_val):
                clazz = Deserializer.__create_class_object(
                    f"{serialized_val[CLASS_MODULE_FIELD]}.{serialized_val[CLASS_NAME_FIELD]}", caller_module)
                kwargs[arg] = Deserializer.__deserialize(clazz, serialized_val[SERIALIZED_FIELD],
                                                         caller_module)
            # TODO: This has to be handled better, but for now only device is a tuple.
            elif type(serialized_val) is tuple:
                kwargs[arg] = torch.device(*serialized_val)
            # TODO: This now also catches functions
            elif is_serialized_type(serialized_val):
                kwargs[arg] = Deserializer.__create_class_object(
                    f"{serialized_val[CLASS_MODULE_FIELD]}.{serialized_val[CLASS_NAME_FIELD]}", caller_module)
            elif is_serialized_collection_of_serializables(serialized_val):
                kwargs[arg] = [
                    Deserializer.__deserialize(
                        Deserializer.__create_class_object(
                            f"{val[CLASS_MODULE_FIELD]}.{val[CLASS_NAME_FIELD]}",
                            caller_module),
                        val[SERIALIZED_FIELD],
                        caller_module)
                    for val in serialized_val]
            # TODO: Won't work if we have functions and callable objects in one collection (works only for functions)
            # TODO: Test this branch
            elif is_serialized_collection_of_callables(serialized_val):
                kwargs[arg] = [
                    Deserializer.__create_class_object(
                        f"{val[CLASS_MODULE_FIELD]}.{val[CLASS_NAME_FIELD]}",
                        caller_module)
                    for val in serialized_val]
            elif is_serialized_primitive(serialized_val) or is_serialized_collection(serialized_val):
                kwargs[arg] = serialized_val

            elif serialized_val is None:
                kwargs[arg] = None
            else:
                raise_unsupported_data_type()

        obj = cls(**{**kwargs, **external_args})
        if package.get('state_dict'):
            obj.load_state_dict(package['state_dict'])

        return (obj, config) if with_config else obj

    @staticmethod
    def __create_class_object(full_class_name: str, caller_module: object):
        # todo: Add import of the package of the class to serialize (if necessary)
        # Creates class object (type) from full_class_name
        # In current module we have only access to top level module (e.g torch)
        # but we want to create particular class.
        # The reduce method with call getattr with more nested path on each iteration
        # (e.g torch, torch.nn, torch.nn.modules, etc.).

        # todo: More generic approach should be implemented. For now only caller's scope and current scope are checked.
        split_path = full_class_name.split(".")
        # todo: change if for a loop
        try:
            return reduce(getattr, split_path, sys.modules[__name__])
        except Exception as e:
            pass

        try:
            return reduce(getattr, split_path, caller_module)
        except Exception as e:
            pass
        # sometimes the next to last step should be removed from class name due to python import syntax
        try:
            return reduce(getattr, split_path[:-2] + split_path[-1:], caller_module)

        except Exception as e:
            return reduce(getattr, split_path[:-2] + split_path[-1:], sys.modules[__name__])


@staticmethod
def __reduce_from_module(full_class_name: str, module: object):
    reduce(getattr, full_class_name.split("."), module)
