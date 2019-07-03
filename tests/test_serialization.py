import os

from torch import nn
from typing import Dict

from model.serializer import ModelSerializer
from model.deserializer import ModelDeserializer

from sonosco.models.deepspeech2_sonosco import DeepSpeech2


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

    model = DeepSpeech2(rnn_type=rnn_type,
                        labels=labels, rnn_hid_size=rnn_hid_size,
                        nb_layers=nb_layers,
                        audio_conf=audio_conf,
                        bidirectional=bidirectional,
                        version=version)

    # serialize
    saver.serialize_model(model, model_path)

    # deserialize
    deserialized_model = loader.deserialize_model(DeepSpeech2, model_path)

    os.remove(model_path)

    # test attributes

    assert deserialized_model.rnn_type == rnn_type
    assert deserialized_model.labels == labels
    assert deserialized_model.rnn_hid_size == rnn_hid_size
    assert deserialized_model.nb_layers == nb_layers
    assert deserialized_model.audio_conf == audio_conf
    assert deserialized_model.bidirectional == bidirectional
    assert deserialized_model.version == version

