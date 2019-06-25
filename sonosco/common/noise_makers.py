import numpy as np

from abc import ABC, abstractmethod


class NoiseMaker(ABC):

    @abstractmethod
    def __call__(self, audio):
        """Adds noise to the audio signal."""
        pass

    def add_noise(self, audio):
        return self(audio)


class GaussianNoiseMaker(NoiseMaker):

    def __init__(self, std=0.002):
        self.std = std

    def __call__(self, audio):
        noise = np.random.randn(len(audio))
        return audio + self.std * noise
