from dataclasses import _process_class, _create_fn, _set_new_attribute, fields, is_dataclass
__primitives = {int, float, str, bool}
__iterables = [list, set, tuple]

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

    # See if we're being called as @dataclass or @dataclass().
    if _cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(_cls)


def __add_serialize(cls):
    fields_to_serialize = fields(cls)
    sonosco_self = ['__sonosco_self__' if 'self' in fields_to_serialize else 'self']
    serialize_body = __create_serialize_body(fields_to_serialize)
    return _create_fn('__serialize__', [sonosco_self], [f'return {serialize_body}'], return_type=dict)


def __create_serialize_body(fields_to_serialize):
    body_lines = ["{"]
    for field in fields_to_serialize:
        if __is_primitive(field) or __is_iterable_of_primitives(field):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}"))
        elif is_dataclass(field.type):
            body_lines.append(__create_dict_entry(field.name, f"self.{field.name}.__serlialize__()"))
        else:
            __throw_unsupported_data_type()
    body_lines.append("}")
    return body_lines


def __is_iterable_of_primitives(field):
    return field.__origin__ in __iterables and field.__args__[0] in __primitives


def __throw_unsupported_data_type():
    raise TypeError("Unsupported data type. Only primitives, lists of primitives, "
                    "@serializable and @dataclass objects can be seralized")


def __create_dict_entry(key, value):
    return f'\'{key}\': {value},'


def __is_primitive(obj):
    return obj.type in __primitives

