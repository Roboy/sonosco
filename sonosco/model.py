import logging
import torch
import deprecation

LOGGER = logging.getLogger(__name__)


class Saver:

    def __init__(self) -> None:
        super().__init__()

    @deprecation.deprecated(
        details="This type of saving may cause problems when path of model class changes. Pleas use save_model instead")
    def save_model_simple(self, model, path):
        """
       Simply saves the model using pickle protocol.
        Args:
            model: model to save
            path (string) : path where to save the model

        Returns:

        """
        torch.save(model, path)

    def save_model(self, model, path, infer_structure=False, serialize_f_name='serialize'):
        """
        Saves the model using pickle protocol.

        If the infer_structure is True this method infers all the meta parameters of the model and save them together
        with learnable parameters.

        If the infer_structure is False and method specified by serialize_f_name exists, the return value of the
        serialize_f_name method is saved.

        If neither of above only learnable parameters a.k.a. state_dict are saved.

        Args:
            model: model to save
            path (string) : path where to save the model
            infer_structure (bool): indicator whether to infer the model structure
            serialize_f_name (string): name of the function that this method should call in order to serialize the model

        Returns:

        """
        entity_to_save = None
        if infer_structure:
            entity_to_save = self.get_constructor_args_with_values(model)
            entity_to_save['state_dict'] = model.state_dict()
        elif hasattr(model, serialize_f_name) and callable(getattr(model, serialize_f_name)):
            entity_to_save = getattr(model, serialize_f_name)()
        else:
            entity_to_save['state_dict'] = model.state_dict()

        torch.save(entity_to_save, path)

    @staticmethod
    def get_constructor_args_with_values(model):
        """
        Assigns values to __init__ params names

        For example:

            class Bar():
                def __init__(self, arg1, arg2):
                    self.arg1 = arg1
                    self.some_other_name = args2


            bar = Bar("A","B")
            get_constructor_args_with_values(bar)
            # returns {arg1: arg1_val, arg2: arg2_val}


        Args:
            model: model to infre from

        Returns (dict): Mapping from __init__ argument to it's value

        """
        return {}


class Loader:

    def load_model(self, cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        cls()
