import numpy as np


class NoiseMaker:

    def __call__(self, audio):
        """Adds noise to the audio signal."""
        pass


class GaussianNoiseMaker(NoiseMaker):

    def __init__(self, std=0.002):
        self.std = std

    def __call__(self, audio):
        noise = np.random.randn(len(audio))
        return audio + self.std * noise
