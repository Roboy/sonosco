from sonosco.inference.asr import SonoscoASR


class DummyASR(SonoscoASR):


    def __init__(self) -> None:
        super().__init__(None)

    def infer(self, sound_bytes):
        return "dummy transcript"
