from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass
class SonoscoAudioInput(ABC):
    """
    Sonosco interface for audio input sources
    """
    @abstractmethod
    def request_audio(self, *args: any, **kwargs: any) -> any: pass
