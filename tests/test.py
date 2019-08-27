import inspect
from typing import Callable, List

from dataclasses import field, fields, dataclass

from sonosco.model.serialization import serializable
from sonosco.model.serializer import ModelSerializer
from sonosco.model.deserializer import ModelDeserializer


def test(arg: Callable[[int, int], float]):
    print(inspect.getsource(arg))


def test2(a: int):
    print(a)


def mydiv(a: str, b: str):
    return int(a) // int(b)


# @serializable
# class CallableClass:
#     some_stuff: str = "XD"
#
#     def __call__(self, *args, **kwargs):
#         return "XDDDD"


@serializable
class MockedNestedClass:
    some_method: Callable
    some_int: int = 5
    some_collection: List[str] = field(default_factory=list)
    # yetAnotherSerializableClass: Callable = CallableClass(some_stuff="XDDDD")


ms = ModelSerializer()

ms.serialize_model(MockedNestedClass(some_method=mydiv), "/Users/w.jurasz/Desktop/serialization_test/ms")
md = ModelDeserializer()
mnc = md.deserialize_model(MockedNestedClass, "/Users/w.jurasz/Desktop/serialization_test/ms")
print(mnc.some_method(10, 5))
# class SomeClass:
#     def __init__(self, method: Callable):
#         self.m = method
#
#     def print_details(self):
