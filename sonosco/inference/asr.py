from abc import ABC, abstractmethod

import torch.nn as nn
from sonosco.datasets import AudioDataProcessor

from sonosco.decoders import GreedyDecoder


class SonoscoASR(ABC):

    # TODO: add processor
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def infer(self, sound_bytes): pass
