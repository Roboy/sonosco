import math
import os

from collections import OrderedDict
from dataclasses import field, dataclass
from torch import nn
from typing import Dict, List, Union, Callable
from sonosco.models.modules import MaskConv, BatchRNN, SequenceWise, InferenceBatchSoftmax
from sonosco.model.serializer import ModelSerializer
from sonosco.model.deserializer import ModelDeserializer
from sonosco.model.serialization import serializable
from abc import ABC, abstractmethod


@serializable
class YetAnotherSerializableClass:
    some_stuff: str = "XD"


@serializable
class MockedNestedClass:
    some_int: int = 5
    some_collection: List[str] = field(default_factory=list)
    yetAnotherSerializableClass: YetAnotherSerializableClass = YetAnotherSerializableClass(some_stuff="XDDDD")


@serializable(model=True, skip_fields=['labels'])
class MockModel(nn.Module):
    mockedNestedClass: MockedNestedClass
    rnn_type: type = nn.LSTM
    labels: str = "abc"
    rnn_hid_size: int = 768
    nb_layers: int = 5
    audio_conf: Dict[str, str] = field(default_factory=dict)
    bidirectional: bool = True
    version: str = '0.0.1'

    def __post_init__(self):
        super(MockModel, self).__init__()
        sample_rate = self.audio_conf.get("sample_rate", 16000)
        window_size = self.audio_conf.get("window_size", 0.02)
        num_classes = len(self.labels)
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_in_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_in_size = int(math.floor(rnn_in_size + 2 * 20 - 41) / 2 + 1)
        rnn_in_size = int(math.floor(rnn_in_size + 2 * 10 - 21) / 2 + 1)
        rnn_in_size *= 32

        rnns = [('0', BatchRNN(input_size=rnn_in_size, hidden_size=self.rnn_hid_size, rnn_type=self.rnn_type,
                               batch_norm=False))]
        rnns.extend(
            [(f"{x + 1}", BatchRNN(input_size=self.rnn_hid_size, hidden_size=self.rnn_hid_size, rnn_type=self.rnn_type))
             for x in range(self.nb_layers - 1)])
        self.rnns = nn.Sequential(OrderedDict(rnns))

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hid_size),
            nn.Linear(self.rnn_hid_size, num_classes, bias=False)
        )

        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )

        self.inference_softmax = InferenceBatchSoftmax()


def test_model_serialization():
    # prepare
    rnn_type: type = nn.GRU
    labels: str = "ABCD"
    rnn_hid_size: int = 256
    nb_layers: int = 10
    audio_conf: Dict[str, str] = {'key': "value"}
    bidirectional: bool = False
    version: str = '1.0.0'

    model_path = "model"

    saver = ModelSerializer()
    loader = ModelDeserializer()

    model = MockModel(rnn_type=rnn_type,
                      labels=labels,
                      rnn_hid_size=rnn_hid_size,
                      nb_layers=nb_layers,
                      audio_conf=audio_conf,
                      bidirectional=bidirectional,
                      version=version,
                      mockedNestedClass=MockedNestedClass(
                          some_int=42,
                          some_collection=['the', 'future', 'is', 'here'],
                          yetAnotherSerializableClass=YetAnotherSerializableClass(some_stuff="old man")))

    # serialize
    saver.serialize_model(model, model_path)

    # deserialize
    deserialized_model = loader.deserialize(MockModel,
                                            model_path,
                                            external_args={
                                                'labels': 'XD12',
                                                'version': '1.0.1'
                                            }
                                            )

    os.remove(model_path)

    # test attributes

    assert len(deserialized_model.state_dict()) == len(model.state_dict())
    assert deserialized_model.state_dict()['conv.seq_module.0.weight'][0][0][0][0] == \
           model.state_dict()['conv.seq_module.0.weight'][0][0][0][0]
    assert deserialized_model.state_dict()['conv.seq_module.0.weight'][0][0][0][1] == \
           model.state_dict()['conv.seq_module.0.weight'][0][0][0][1]
    assert deserialized_model.state_dict()['conv.seq_module.0.weight'][0][0][0][5] == \
           model.state_dict()['conv.seq_module.0.weight'][0][0][0][5]
    assert deserialized_model.rnn_type == rnn_type
    assert deserialized_model.labels == 'XD12'
    assert deserialized_model.rnn_hid_size == rnn_hid_size
    assert deserialized_model.nb_layers == nb_layers
    assert deserialized_model.audio_conf == audio_conf
    assert deserialized_model.bidirectional == bidirectional
    assert deserialized_model.version == '1.0.1'
    assert deserialized_model.mockedNestedClass.some_int == 42
    assert deserialized_model.mockedNestedClass.some_collection == ['the', 'future', 'is', 'here']
    assert deserialized_model.mockedNestedClass.yetAnotherSerializableClass.some_stuff == "old man"


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
class TestClass:
    some_method: Callable[[str, str], int]
    some_int: int = 5
    some_collection: List[str] = field(default_factory=list)
    yetAnotherSerializableClass: Union[Callable[[any], any], Callable[[int], int]] = SubClass1(some_stuff="XDDDD")
    serializable_list: List[CallableClass] = field(default_factory=list)


def some_method_other():
    return "value"


def test_model_serialization2():
    # prepare
    model_path = "model"
    saver = ModelSerializer()
    loader = ModelDeserializer()
    testClass = TestClass(
        some_method=some_method_other,
        serializable_list=[SubClass1("some other stuff"), SubClass2()])

    # serialize
    saver.serialize_model(testClass, model_path)

    # deserialize
    deserialized_class = loader.deserialize(TestClass, model_path)

    os.remove(model_path)

    # test attributes

    assert deserialized_class.some_int == 5
    assert deserialized_class.some_collection == []
    assert deserialized_class.some_method == some_method_other
    assert deserialized_class.some_method() == some_method_other()
    assert deserialized_class.yetAnotherSerializableClass.__class__ == SubClass1
    assert deserialized_class.yetAnotherSerializableClass.some_stuff == "XDDDD"
    assert len(deserialized_class.serializable_list) == 2
    assert deserialized_class.serializable_list[0].__class__ == SubClass1
    assert deserialized_class.serializable_list[0]() == "some other stuff"
    assert deserialized_class.serializable_list[1].__class__ == SubClass2
    assert deserialized_class.serializable_list[1]() == "Calling class2"


test_model_serialization()
