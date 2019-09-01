from abc import ABC, abstractmethod

import torch.nn as nn
from sonosco.datasets import AudioDataProcessor
from sonosco.model.deserializer import ModelDeserializer
from sonosco.decoders import GreedyDecoder


class SonoscoASR(ABC):

    # TODO: add processor
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.loader = ModelDeserializer()

    @abstractmethod
    def infer(self, sound_bytes): pass
