from sonosco.inference.asr import SonoscoASR


class DummyASR(SonoscoASR):

    def __init__(self) -> None:
        """
        Dummy implementation of ASR with fixed return values
        """
        super().__init__("")

    def infer(self, sound_bytes: any) -> str:
        """

        Args:
            sound_bytes:

        Returns: dummy transcript"

        """
        return "dummy transcript"

    def infer_from_path(self, path: str) -> str:
        """

        Args:
            path:

        Returns: "dummy transcript"

        """
        return "dummy transcript"
