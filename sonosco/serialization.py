from dataclasses import _process_class, _create_fn, _set_new_attribute, fields, is_dataclass

__primitives = {int, float, str, bool}
# TODO support for dict with Union value type
__iterables = [list, set, tuple, dict]


def serializable(_cls=None):
    """

    Returns the same class as was passed in, with init and serialize methods.


    Args:
        _cls:

    Returns:

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


def is_serializable(obj):
    return hasattr(obj, '__serialize__')


def __add_serialize(cls):
    fields_to_serialize = fields(cls)
    sonosco_self = '__sonosco_self__' if 'self' in fields_to_serialize else 'self'
    serialize_body = __create_serialize_body(fields_to_serialize)
    return _create_fn('__serialize__', [sonosco_self], serialize_body, return_type=dict)


def __create_serialize_body(fields_to_serialize):
    body_lines = ["return {"]
    for field in fields_to_serialize:
        if __is_primitive(field.type) or __is_iterable_of_primitives(field):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}"))
        elif is_dataclass(field.type):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}.__serlialize__()"))
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


def __extract_from_nn(name, cls, body_lines):
    constants = list(filter(lambda el: not el.startswith('_'), cls.__constants__))
    for constant in constants:
        body_lines.append(__create_dict_entry(constant, f"self.{name}.{constant}"))


def __is_iterable_of_primitives(field):
    return hasattr(field.type, '__origin__') and field.type.__origin__ in __iterables and __is_primitive(
        field.type.__args__[0])


def __throw_unsupported_data_type():
    raise TypeError("Unsupported data type. Only primitives, lists of primitives, types"
                    "and @serializable objects can be serialized")


def __create_dict_entry(key, value):
    return f'\'{key}\': {value},'


def __is_primitive(obj):
    return all(el in __primitives for el in obj) if obj is tuple else obj in __primitives


def __is_type(cls):
    return cls is type
