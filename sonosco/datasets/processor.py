import logging
import torch
import librosa
import numpy as np
import scipy.signal
import sonosco.common.audio_tools as audio_tools
import sonosco.common.utils as utils
import sonosco.common.noise_makers as noise_makers

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

LOGGER = logging.getLogger(__name__)
MIN_STRETCH = 0.7
MAX_STRETCH = 1.3
MIN_PITCH = 0.7
MAX_PITCH = 1.5
MAX_SHIFT = 4000


class AudioDataProcessor:

    def __init__(self, window_stride, window_size, sample_rate, labels="abc", normalize=False, augment=False, **kwargs):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param window_stride: number of seconds to skip between each window
        :param window_size: number of seconds to use for a window of spectrogram
        :param sample_rate: sample rate of the recordings
        :param labels: string containing all the possible characters to map to
        :param normalize: apply standard mean and deviation normalization to audio tensor
        :param augment(default False): apply random tempo and gain perturbations
        """
        self.window_stride = window_stride
        self.window_size = window_size
        self.window = windows.get(kwargs['window'], windows['hamming'])
        self.sample_rate = sample_rate
        self.labels_map = utils.labels_to_dict(labels)
        self.normalize = normalize
        self.augment = augment

    @property
    def window_stride_samples(self):
        return int(self.sample_rate * self.window_stride)

    @property
    def window_size_samples(self):
        return int(self.sample_rate * self.window_size)

    def retrieve_file(self, audio_path):
        sound, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        return sound, sample_rate

    def augment_audio(self, sound, stretch=True, shift=False, pitch=True, noise=True):
        augmented = audio_tools.stretch(sound, utils.random_float(MIN_STRETCH, MAX_STRETCH)) if stretch else sound
        augmented = audio_tools.shift(augmented, np.random.randint(MAX_SHIFT)) if shift else augmented
        augmented = audio_tools.pitch_shift(augmented, self.sample_rate,
                                            n_steps=utils.random_float(MIN_PITCH, MAX_PITCH)) if pitch else augmented

        if noise:
            noise_maker = noise_makers.GaussianNoiseMaker()
            augmented = noise_maker.add_noise(augmented) if noise else augmented

        return augmented

    def parse_audio_from_file(self, audio_path, raw=False):
        sound, sample_rate = self.retrieve_file(audio_path)
        if raw:
            return sound

        spectogram = self.parse_audio(sound, sample_rate)

        return spectogram

    def parse_audio(self, sound, sample_rate):
        if sample_rate != self.sample_rate:
            raise ValueError(f"The stated sample rate {self.sample_rate} and the factual rate {sample_rate} differ!")

        if self.augment:
            sound = self.augment_audio(sound)

        complex_spectrogram = librosa.stft(sound,
                                           n_fft=self.window_size_samples,
                                           hop_length=self.window_stride_samples,
                                           win_length=self.window_size_samples,
                                           window=self.window)
        spectrogram, phase = librosa.magphase(complex_spectrogram)
        # S = log(S+1)
        spectrogram = np.log1p(spectrogram)
        spectrogram = torch.FloatTensor(spectrogram)

        if self.normalize:
            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram.add_(mean)
            spectrogram.div_(std)

        return spectrogram

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        # TODO: Is it fast enough?
        transcript = list(filter(lambda el: el is not None,
                                 [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def parse_audio_for_inference(self, audio_path):
        """
        Return spectrogram and its length in a format used for inference.
        :param audio_path: Audio path.
        :return: spect [1, seq_length, freqs], lens [scalar]
        """
        spect = self.parse_audio_from_file(audio_path)
        spect = spect.view(1, spect.size(0), spect.size(1)).transpose(1, 2)
        lens = torch.IntTensor([spect.shape[1]]).int()
        return spect, lens
