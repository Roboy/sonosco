from sonosco.inference.asr import SonoscoASR


class DummyASR(SonoscoASR):

    def __init__(self) -> None:
        super().__init__("")

    def infer(self, sound_bytes):
        return "dummy transcript"

    def infer_from_path(self, path: str) -> str:
        return "dummy transcript"
