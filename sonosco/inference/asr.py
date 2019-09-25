import tempfile

from abc import ABC, abstractmethod
from sonosco.serialization import Deserializer


class SonoscoASR(ABC):

    def __init__(self, model_path: str) -> None:
        """
        Sonosco interface for Automatic speech recognition
        Args:
            model_path: path to model used in recognition
        """
        super().__init__()
        self.model_path = model_path
        self.loader = Deserializer()

    def infer(self, audio: any) -> str:
        """
        Infer speech from audio
        Args:
            audio: audio to infer

        Returns: inferred text

        """
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(audio)
        return self.infer_from_path(temp_audio_file.name)

    @abstractmethod
    def infer_from_path(self, path: str) -> str:
        """
        Infer speech from path
        Args:
            path: path to audio

        Returns: infered text

        """
        pass
