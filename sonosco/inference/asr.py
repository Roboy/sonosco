import tempfile

from abc import ABC, abstractmethod
from sonosco.model.deserializer import Deserializer


class SonoscoASR(ABC):

    # TODO: add processor
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.loader = Deserializer()

    def infer(self, audio) -> str:
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(audio)
        return self.infer_from_path(temp_audio_file.name)

    @abstractmethod
    def infer_from_path(self, path: str) -> str:
        pass
