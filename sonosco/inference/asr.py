from abc import ABC, abstractmethod

from dataclasses import dataclass
import torch.nn as nn


@dataclass
class SonoscoASR(ABC):
    model: nn.Module

    @abstractmethod
    def infer(self, sound_bytes): pass
