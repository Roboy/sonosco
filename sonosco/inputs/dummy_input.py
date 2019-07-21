from sonosco.inputs.audio import SonoscoAudioInput


class DummyInput(SonoscoAudioInput):

    def request_audio(self, *args, **kwargs):
        return None
