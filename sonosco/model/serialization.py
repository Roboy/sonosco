import collections

from sonosco.common.constants import CLASS_MODULE_FIELD, CLASS_NAME_FIELD, SERIALIZED_FIELD
from dataclasses import _process_class, _create_fn, _set_new_attribute, fields
import typing
from torch import nn
import torch

__primitives = {int, float, str, bool}
# TODO support for dict with Union value type
__iterables = [list, set, tuple, dict]


# TODO: Prevent user from serializing lambdas.
# Only named methods can be serialized or the function has to be shipped separately

# TODO: Some errors might not be found at annotating time (e.g. wrong callable or model not being serializable)
# Thus it would be good to run "dry" serialization before user runs it after training.

def serializable(_cls: type = None, *, model=False, enforced_serializable: list = None,
                 skip_fields: list = None) -> object:
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
        _cls type: Python Class object
        model bool: indicator whether class is a Pytorch model

    Returns: enchanted class

    """
    if enforced_serializable is None:
        enforced_serializable = []

    if skip_fields is None:
        skip_fields = []

    def wrap(cls):
        cls = _process_class(cls, init=True, repr=False, eq=False, order=False, unsafe_hash=False, frozen=False)
        _set_new_attribute(cls, '__serialize__', __add_serialize(cls, model, enforced_serializable, skip_fields))
        return cls

    # See if we're being called as @serializable or @serializable().
    if _cls is None:
        # We're called with parens.
        return wrap

    # We're called as @serializable without parens.
    return wrap(_cls)


def is_serializable(obj: object) -> bool:
    """
    Checks if object is serializable.
    Args:
        obj: object to test

    Returns (bool): serializable indicator

    """
    return hasattr(obj, '__serialize__')


def __add_serialize(cls: type, model: bool, enforced_serializable: list, skip_fields: list) -> object:
    """
    Adds __serialize__ method.
    Args:
        cls: class to enchant

    Returns (object): __serialize__ method body

    """
    fields_to_serialize = list(filter(lambda el: el.name not in skip_fields, fields(cls)))
    sonosco_self = '__sonosco_self__' if 'self' in fields_to_serialize else 'self'
    serialize_body = __create_serialize_body(fields_to_serialize, model, enforced_serializable)
    return _create_fn('__serialize__', [sonosco_self], serialize_body, return_type=dict)


def __create_serialize_body(fields_to_serialize: typing.Iterable, model: bool, enforced_serializable: list) -> \
        typing.List[str]:
    """
    Creates body of __serialize__ method as list of strings.
    Args:
        fields_to_serialize (Iterable): iterable of fields to serialize

    Returns (list): __serialize__ method body as list of strings

    """

    fields_to_serialize = fields_to_serialize
    callables = set(filter(lambda el: __is_callable(el.type), fields_to_serialize))
    serializable_iterables = set(filter(lambda el: __is_iterable_of_serializables(el.type), fields_to_serialize))
    callable_iterables = set(filter(lambda el: __is_iterable_of_callables(el.type), fields_to_serialize))

    body_lines = ["from types import FunctionType"]
    body_lines.append("from sonosco.model.serialization import is_serializable")

    for c in callables:
        body_lines.append(f"if isinstance(self.{c.name}, FunctionType):")
        body_lines.append(f"    {c.name} = {{")
        __encode_type_serialization(body_lines, c.name)
        body_lines.append(f"}}")
        body_lines.append(f"elif is_serializable(self.{c.name}):")
        body_lines.append(f"    {c.name} = {{")
        __encode_serializable_serialization(body_lines, c)
        body_lines.append(f"}}")
        body_lines.append(f"else: raise TypeError(\"Callable must be a function for now\")")

    for field in callable_iterables:
        body_lines.append(f"{field.name} = []")
        body_lines.append(f"for el in self.{field.name}:")
        body_lines.append(f"    if isinstance(el, FunctionType):")
        body_lines.append(f"        tmp = {{")
        __encode_type_serialization(body_lines, "el", False)
        body_lines.append(f"        }}")
        body_lines.append(f"    elif is_serializable(el):")
        body_lines.append(f"        tmp = {{")
        __encode_serializable_serialization(body_lines, field)
        body_lines.append(f"        }}")
        body_lines.append(f"    else: raise TypeError(\"Callable must be a function or @serializable class for now\")")
        body_lines.append(f"    {field.name}.append(tmp)")

    for field in serializable_iterables:
        body_lines.append(f"{field.name} = []")
        body_lines.append(f"for el in self.{field.name}:")
        body_lines.append(f"    {field.name}.append({{")
        body_lines.append(__create_dict_entry(CLASS_NAME_FIELD, f"el.__class__.__name__"))
        body_lines.append(__create_dict_entry(CLASS_MODULE_FIELD, f"el.__class__.__module__"))
        body_lines.append(__create_dict_entry(SERIALIZED_FIELD, f"el.__serialize__()"))
        body_lines.append("})")

    body_lines.append("return {")
    for field in fields_to_serialize:
        # TODO: Rewrite this ugly if else chain to something more OO
        if __is_primitive(field.type) or __is_iterable_of_primitives(field.type):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}"))
        elif field.type == torch.device:
            body_lines.append(__create_dict_entry(field.name, f"(self.{field.name}.type, self.{field.name}.index)"))
        elif is_serializable(field.type) or field.name in enforced_serializable:
            body_lines.append(f"'{field.name}': {{")
            __encode_serializable_serialization(body_lines, field)
            body_lines.append("},")
        elif __is_iterable_of_serializables(field.type) or \
                __is_iterable_of_callables(field.type) or \
                __is_callable(field.type):
            body_lines.append(f"'{field.name}': {field.name},")
        elif __is_type(field.type):
            body_lines.append(f"'{field.name}': {{")
            __encode_type_serialization(body_lines, field.name)
            body_lines.append("},")
        else:
            __throw_unsupported_data_type(field)
    if model:
        body_lines.append(__create_dict_entry("state_dict", "self.state_dict()"))
    body_lines.append("}")
    return body_lines


def __encode_type_serialization(body_lines, name, with_self=True):
    s = "self." if with_self else ""
    body_lines.append(__create_dict_entry(CLASS_NAME_FIELD, f"{s}{name}.__name__"))
    body_lines.append(__create_dict_entry(CLASS_MODULE_FIELD, f"{s}{name}.__module__"))


def __encode_serializable_serialization(body_lines, field):
    body_lines.append(__create_dict_entry(CLASS_NAME_FIELD, f"self.{field.name}.__class__.__name__"))
    body_lines.append(__create_dict_entry(CLASS_MODULE_FIELD, f"self.{field.name}.__class__.__module__"))
    body_lines.append(__create_dict_entry(SERIALIZED_FIELD, f"self.{field.name}.__serialize__()"))


def __extract_from_nn(name: str, cls: nn.Module, body_lines: typing.List[str]):
    """
    Extract fields from torch.nn.Module class and updated body of __serialize methods with them.

    Args:
        name: name of the instance of nn.Module in containing class
        cls: class object of nn.Module
        body_lines: current body of __serialize__ method

    Returns:

    """
    constants = list(filter(lambda el: not el.startswith('_'), cls.__constants__))
    for constant in constants:
        body_lines.append(__create_dict_entry(constant, f"self.{name}.{constant}"))


def __is_iterable_of_primitives(field) -> bool:
    return __is_iterable(field) and hasattr(field, '__args__') and __is_primitive(field.__args__[0])


def __is_iterable_of_serializables(field) -> bool:
    return __is_iterable(field) and hasattr(field, '__args__') and is_serializable(field.__args__[0])


def __is_iterable_of_callables(field) -> bool:
    return __is_iterable(field) and hasattr(field, '__args__') and __is_callable(field.__args__[0])


def __is_iterable(field) -> bool:
    return hasattr(field, '__origin__') and field.__origin__ in __iterables


def __throw_unsupported_data_type(field):
    """
    Throws unsupported data type exception.
    Returns:

    """
    raise TypeError(f"Field name: {field.name}, Field type: {field.type}. "
                    f"Unsupported data type. Currently only primitives, lists of primitives, types, callables "
                    f"and other serializable objects can be serialized")


def __create_dict_entry(key: str, value: str) -> str:
    """
    Creates a dict entry in for of a string: {'key': value}
    Args:
        key (str): key of entry
        value (str): value of entry

    Returns (str): dict entry

    """
    return f'\'{key}\': {value},'


def __is_primitive(obj: any) -> bool:
    """
    Checks if object is Python primitive.
    Args:
        obj (any): object to check

    Returns (bool): indicator of obj is primitive

    """
    return all(el in __primitives for el in obj) if obj is tuple else obj in __primitives


def __is_type(obj: any) -> bool:
    """
    Checks if obj is Python class.
    Args:
        obj (any): object to check

    Returns (bool): indicator of obj is class

    """
    return obj is type


def __is_callable(obj: any) -> bool:
    """
    Check if object is a collections.abc.Callable or typing.Union of Callable
    Args:
         obj (any): object to check

    Returns (bool): indicator of obj is a collections.abc.Callable or typing.Union of Callable

    """

    # TODO: Extend support for Unions to all other types
    if hasattr(obj, '__origin__') and obj.__origin__ == typing.Union:
        objs = list(obj.__args__)
    else:
        objs = [obj]
    return all([hasattr(obj, '__origin__') and obj.__origin__ == collections.abc.Callable for obj in objs])
