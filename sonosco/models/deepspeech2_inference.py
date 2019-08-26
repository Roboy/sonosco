from sonosco.inference.asr import SonoscoASR
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.models.deepspeech2 import DeepSpeech2
import tempfile
import torch
import librosa

class DeepSpeech2Inference(SonoscoASR):

    def __init__(self, model, cuda=False):
        super().__init__(model)
        self.processor = AudioDataProcessor(**model.audio_conf, normalize=True)
        self.decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
        self.cuda = cuda

    def infer(self, audio):
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(audio)
        loaded, sr = librosa.load(temp_audio_file.name, sr=self.processor.sample_rate)
        spect = self.processor.parse_audio(sound=loaded, sample_rate=sr)
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        spect = spect.to(torch.device("cuda" if self.cuda else "cpu"))
        input_sizes = torch.IntTensor([spect.size(3)]).int()
        out, output_sizes = self.model(spect, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        return decoded_output[0]
