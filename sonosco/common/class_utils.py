import inspect
from typing import Set

__primitives = {int, float, str, bool}


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


def is_collection(field):
    pass


def is_primitive(obj):
    return all(el in __primitives for el in obj) if obj is tuple else obj in __primitives


def is_type(cls):
    pass

def throw_unsupported_data_type():
    raise TypeError("Unsupported data type. Currently only primitives, lists of primitives and types"
                    "objects can be serialized")
