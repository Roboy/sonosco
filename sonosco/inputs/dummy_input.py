from sonosco.inputs.audio import SonoscoAudioInput


class DummyInput(SonoscoAudioInput):
    """
    Dummy implementation, returns None
    """
    def request_audio(self, *args: any, **kwargs: any) -> any:
        """

        Args:
            *args:
            **kwargs:

        Returns: None

        """
        return None
