import inspect
from typing import Set

__primitives = {int, float, str, bool}
__collections = [list, set, tuple, dict]

def get_constructor_args(cls) -> Set[str]:
    """
    E.g.

        class Bar():
                def __init__(self, arg1, arg2):

        get_constructor_args(Bar)
        # returns ['arg1', 'arg2']
    Args:
        cls (object):

    Returns: set containing names of constructor arguments

    """
    return set(inspect.getfullargspec(cls.__init__).args[1:])


def get_class_by_name(name: str) -> type:
    """
    Returns type object of class specified by name
    Args:
        name: full name of the class (with packages)

    Returns: class object

    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def is_serialized_collection(obj):
    return type(obj) in __collections


def is_serialized_primitive(obj):
    return type(obj) in __primitives


def is_serialized_type(obj):
    return type(obj) is dict and '__class_module' in obj and '__class_name' in obj

def throw_unsupported_data_type():
    raise TypeError("Unsupported data type. Currently only primitives, lists of primitives and types"
                    "objects can be serialized")
