from dataclasses import _process_class, _create_fn, _set_new_attribute, fields

from typing import List

from torch import nn

__primitives = {int, float, str, bool}
# TODO support for dict with Union value type
__iterables = [list, set, tuple, dict]


def serializable(_cls: type = None) -> object:
    """

    Returns the same class as passed in, but with __init__ and __serialize__ methods.
    This annotation is necessary to use sonosco (de)serialization.
    This annotation has following requirements:
        * All object parameters have to be defined on the class level.
        * All the parameters have to have type explicitly defined (by python primitives or typing library)
        * Class has to extend torch.nn.Module class
        * Currently only primitives, lists of primitives and types objects are supported.

    __init__ method assigns arguments to parameters (keeping the same name)
    __serialize__ method returns dictionary containing state of the object and torch state_dict


    Args:
        _cls: Python Class object

    Returns: enchanted class

    """

    def wrap(cls):
        cls = _process_class(cls, init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=False)
        _set_new_attribute(cls, '__serialize__', __add_serialize(cls))
        return cls

    # See if we're being called as @serializable or @serializable().
    if _cls is None:
        # We're called with parens.
        return wrap

    # We're called as @serializable without parens.
    return wrap(_cls)


def is_serializable(obj: object) -> bool:
    """
    Checks if object is serializable
    Args:
        obj: object to test

    Returns (bool): serializable indicator

    """
    return hasattr(obj, '__serialize__')


def __add_serialize(cls: type) -> object:
    """
    Adds __serialize__ method
    Args:
        cls: class to enchant

    Returns (object): __serialize__ method body

    """
    fields_to_serialize = fields(cls)
    sonosco_self = '__sonosco_self__' if 'self' in fields_to_serialize else 'self'
    serialize_body = __create_serialize_body(fields_to_serialize)
    return _create_fn('__serialize__', [sonosco_self], serialize_body, return_type=dict)


def __create_serialize_body(fields_to_serialize: list) -> List[str]:
    """
    Creates body of __serialize__ method as list of strings
    Args:
        fields_to_serialize: list of fields to serialize

    Returns (list): __serialize__ method body as list of strings

    """
    body_lines = ["return {"]
    for field in fields_to_serialize:
        # TODO: Rewrite this ugly if else chain to something more OO
        if __is_primitive(field.type) or __is_iterable_of_primitives(field.type):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}"))
        # TODO: add deserialization of classes
        # elif is_dataclass(field.type):
        #     body_lines.append(__create_dict_entry(field.name, f"self.{field.name}.__serlialize__()"))
        elif __is_type(field.type):
            body_lines.append(f"'{field.name}': {{")
            body_lines.append(__create_dict_entry("__class_name", f"self.{field.name}.__name__"))
            body_lines.append(__create_dict_entry("__class_module", f"self.{field.name}.__module__"))
            body_lines.append("},")
        else:
            __throw_unsupported_data_type()
    body_lines.append(__create_dict_entry("state_dict", "self.state_dict()"))
    body_lines.append("}")
    return body_lines


def __extract_from_nn(name: str, cls: nn.Module, body_lines: List[str]):
    """
    Extract fields from torch.nn.Module class and updated body of __serialize methods with them,

    Args:
        name: name of the instance of nn.Module in containing class
        cls: class object of nn.Module
        body_lines: current body of __serialize__ method

    Returns:

    """
    constants = list(filter(lambda el: not el.startswith('_'), cls.__constants__))
    for constant in constants:
        body_lines.append(__create_dict_entry(constant, f"self.{name}.{constant}"))


def __is_iterable_of_primitives(field: type) -> bool:
    return hasattr(field, '__origin__') and field.__origin__ in __iterables and __is_primitive(
        field.__args__[0])


def __throw_unsupported_data_type():
    """
    Throws unsupported data type exception
    Returns:

    """
    raise TypeError("Unsupported data type. Currently only primitives, lists of primitives and types"
                    "objects can be serialized")


def __create_dict_entry(key: str, value: str) -> str:
    """
    Creates a dict entry in for of a string: {'key': value}
    Args:
        key (str): key of entry
        value (str): value of entry

    Returns (str): dict entry

    """
    return f'\'{key}\': {value},'


def __is_primitive(obj: object) -> bool:
    """
    Checks if object is Python primitive
    Args:
        obj (object): object to check

    Returns (bool): indicator of obj is primitive

    """
    return all(el in __primitives for el in obj) if obj is tuple else obj in __primitives


def __is_type(obj):
    """
    Checks if object is Python class
    Args:
        obj (object): object to check

    Returns (bool): indicator of obj is class

    """
    return obj is type
