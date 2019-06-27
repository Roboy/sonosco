import os

from torch import nn
from typing import Dict

from model import Saver, Loader

from sonosco.models.deepspeech2_sonosco import DeepSpeech2


# @pytest.fixture
# # def logger():
# #     logger = logging.getLogger(SONOSCO)
# #     setup_logging(logger)
# #     return logger


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

    saver = Saver()
    loader = Loader()

    model = DeepSpeech2(rnn_type=rnn_type,
                        labels=labels, rnn_hid_size=rnn_hid_size,
                        nb_layers=nb_layers,
                        audio_conf=audio_conf,
                        bidirectional=bidirectional,
                        version=version)

    # serialize
    saver.save_model(model, model_path)

    # deserialize
    deserialized_model = loader.load_model(DeepSpeech2, model_path)

    os.remove(model_path)

    # test attributes

    assert deserialized_model.rnn_type == rnn_type
    assert deserialized_model.labels == labels
    assert deserialized_model.rnn_hid_size == rnn_hid_size
    assert deserialized_model.nb_layers == nb_layers
    assert deserialized_model.audio_conf == audio_conf
    assert deserialized_model.bidirectional == bidirectional
    assert deserialized_model.version == version
    # assert deserialized_model.state_dict() == model.state_dict()


test_model_serialization()
