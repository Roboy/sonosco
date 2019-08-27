import inspect
from typing import Callable, List, Union

from dataclasses import field, fields, dataclass

from sonosco.model.serialization import serializable
from sonosco.model.serializer import ModelSerializer
from sonosco.model.deserializer import ModelDeserializer
from abc import ABC, abstractmethod
from sonosco.training.trainer import ModelTrainer

@serializable
class CallableClass(ABC):
    some_stuff: str = "XD"

    @abstractmethod
    def __call__(self, *args, **kwargs):
        return "XDDDD"

@serializable()
class SubClass1(CallableClass):
    some_stuff: str = "Calling class1"

    def __call__(self, *args, **kwargs):
        return self.some_stuff


@serializable()
class SubClass2(CallableClass):
    some_stuff: str = "Calling class2"

    def __call__(self, *args, **kwargs):
        return self.some_stuff


@serializable
class MockedNestedClass:
    some_method: Callable[[str, str], int]
    some_int: int = 5
    some_collection: List[str] = field(default_factory=list)
    yetAnotherSerializableClass: Union[Callable[[any], any], Callable[[int], int]] = SubClass1(some_stuff="XDDDD")
    lists: List[CallableClass] = field(default_factory=list)

    def execute_callables(self):
        for el in self.lists:
            print(el())
        return str(self.some_method(10, 5)) + self.yetAnotherSerializableClass()


ms = ModelSerializer()
mt = ModelTrainer()
mt = ms.serialize_model(mt, "/Users/w.jurasz/Desktop/serialization_test/ms")

md = ModelDeserializer()
mnc = md.deserialize_model(ModelTrainer, "/Users/w.jurasz/Desktop/serialization_test/ms")

