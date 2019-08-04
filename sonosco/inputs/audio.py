from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class SonoscoAudioInput(ABC):

    @abstractmethod
    def request_audio(self, *args, **kwargs): pass