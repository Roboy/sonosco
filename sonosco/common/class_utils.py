import inspect
from typing import List


def get_constructor_args(cls: type) -> List[str]:
    """
    E.g.

        class Bar():
                def __init__(self, arg1, arg2):

        get_constructor_args(BAR)
        # returns ['arg1', 'arg2']
    Args:
        cls (type):

    Returns: list containing names of constructor arguments

    """
    return inspect.getfullargspec(cls.__init__).args[1:]


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
