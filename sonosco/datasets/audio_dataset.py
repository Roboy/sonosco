# ----------------------------------------------------------------------------
# Based on SeanNaren's deepspeech.pytorch:
# https://github.com/SeanNaren/deepspeech.pytorch
# ----------------------------------------------------------------------------

import warnings
import os
import logging
import torch
import torchaudio
import sonosco.config.global_settings as global_settings

from typing import Tuple
from torch.utils.data import Dataset
from sonosco.common.utils import setup_logging
from sonosco.common.constants import *


LOGGER = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, window_stride, window_size, sample_rate, labels="abc", normalize=False, augment=False):
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

    @staticmethod
    def retrieve_file(audio_path):
        sound, sample_rate = torchaudio.load(audio_path)
        return sound, sample_rate

    @staticmethod
    def augment_audio(sound, tempo_range: Tuple = (0.85, 1.15), gain_range: Tuple = (-6, 8)):
        """Changes tempo and gain of the wave."""
        warnings.warn("Augmentation is not implemented")  # TODO: Implement
        return sound

    def parse_audio(self, audio_path):
        sound, sample_rate = self.retrieve_file(audio_path)

        if sample_rate != self.sample_rate:
            raise ValueError(f"The stated sample rate {self.sample_rate} and the factual rate {sample_rate} differ!")

        if self.augment:
            sound = self.augment_audio(sound)

        if global_settings.CUDA_ENABLED:
            sound = sound.cuda()

        # TODO: comment why take the last element?
        spectrogram = torch.stft(torch.from_numpy(sound.numpy().T.squeeze()),
                                 n_fft=self.window_size_samples,
                                 hop_length=self.window_stride_samples,
                                 win_length=self.window_size_samples,
                                 window=torch.hamming_window(self.window_size_samples),
                                 normalized=self.normalize)[:, :, -1]

        return spectrogram

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
            LOGGER.info(f"1: {transcript}")
        # TODO: Is it fast enough?
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        LOGGER.info(f"transcript_path: {transcript_path} transcript: {transcript}")
        return transcript


class AudioDataset(Dataset):

    def __init__(self, processor: DataProcessor, manifest_filepath):
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

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        spectrogram = self.processor.parse_audio(audio_path)
        transcript = self.processor.parse_transcript(transcript_path)

        return spectrogram, transcript

    def __len__(self):
        return self.size
