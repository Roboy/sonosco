# ----------------------------------------------------------------------------
# Based on SeanNaren's deepspeech.pytorch:
# https://github.com/SeanNaren/deepspeech.pytorch
# ----------------------------------------------------------------------------

import logging
import torch
import librosa
import numpy as np
import sonosco.config.global_settings as global_settings
import sonosco.common.audio_tools as audio_tools
import sonosco.common.utils as utils

from torch.utils.data import Dataset


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
        self.sample_rate = sample_rate
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.normalize = normalize
        self.augment = augment

    @property
    def window_stride_samples(self):
        return int(self.sample_rate * self.window_stride)

    @property
    def window_size_samples(self):
        return int(self.sample_rate * self.window_stride)

    def retrieve_file(self, audio_path):
        sound, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        return sound, sample_rate

    def augment_audio(self, sound, stretch=True, shift=False, pitch=True, noise=True):
        augmented = audio_tools.stretch(sound, utils.random_float(MIN_STRETCH, MAX_STRETCH)) if stretch else sound
        augmented = audio_tools.shift(augmented, np.random.randint(MAX_SHIFT)) if shift else augmented
        augmented = audio_tools.pitch_shift(augmented, self.sample_rate,
                                            n_steps=utils.random_float(MIN_PITCH, MAX_PITCH)) if pitch else augmented
        augmented = audio_tools.add_noise(augmented) if noise else augmented
        return augmented

    def parse_audio(self, audio_path, raw=False):
        sound, sample_rate = self.retrieve_file(audio_path)

        if sample_rate != self.sample_rate:
            raise ValueError(f"The stated sample rate {self.sample_rate} and the factual rate {sample_rate} differ!")

        if self.augment:
            sound = self.augment_audio(sound)

        if raw:
            return sound

        # TODO: comment why take the last element?
        complex_spectrogram = librosa.stft(sound,
                                           n_fft=self.window_size_samples,
                                           hop_length=self.window_stride_samples,
                                           win_length=self.window_size_samples)
        spectrogram, phase = librosa.magphase(complex_spectrogram)
        # S = log(S+1)
        spectrogram = torch.from_numpy(np.log1p(spectrogram))

        return spectrogram

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        # TODO: Is it fast enough?
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        LOGGER.debug(f"transcript_path: {transcript_path} transcript: {transcript}")
        return transcript


class AudioDataset(Dataset):

    def __init__(self, processor: AudioDataProcessor, manifest_filepath):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param processor: Data processor object
        :param manifest_filepath: Path to manifest csv as describe above
        """
        super().__init__()
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.processor = processor

    def get_raw(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        sound = self.processor.parse_audio(audio_path, raw=True)
        transcript = self.processor.parse_transcript(transcript_path)

        return sound, transcript

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        spectrogram = self.processor.parse_audio(audio_path)
        transcript = self.processor.parse_transcript(transcript_path)

        return spectrogram, transcript

    def __len__(self):
        return self.size
