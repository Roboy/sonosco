from sonosco.inference.asr import SonoscoASR
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.models.deepspeech2 import DeepSpeech2
import torch

class DeepSpeech2Inference(SonoscoASR):

    def __init__(self, model):
        super().__init__()
        self.processor = AudioDataProcessor(**model.audio_conf, normalize=True)
        self.decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    def infer(self, audio):
        with open("audio.wav", "wb") as file:
            file.write(audio)
        spect = self.processor.parse_audio("audio.wav")
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        spect = spect.to(torch.device("cuda" if self.cuda else "cpu"))
        input_sizes = torch.IntTensor([spect.size(3)]).int()
        out, output_sizes = self.model(spect, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        return decoded_output[0]
