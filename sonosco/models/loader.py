import logging
import torch
import deprecation
import torch.nn as nn

from sonosco.common.class_utils import get_constructor_args, get_class_by_name

LOGGER = logging.getLogger(__name__)


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
