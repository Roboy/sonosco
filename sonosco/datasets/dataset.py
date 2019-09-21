# ----------------------------------------------------------------------------
# Based on SeanNaren's deepspeech.pytorch:
# https://github.com/SeanNaren/deepspeech.pytorch
# ----------------------------------------------------------------------------

import logging

from torch.utils.data import Dataset
from .processor import AudioDataProcessor


LOGGER = logging.getLogger(__name__)


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

        sound = self.processor.parse_audio_from_file(audio_path, raw=True)
        transcript = self.processor.parse_transcript(transcript_path)

        return sound, transcript

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        spectrogram = self.processor.parse_audio_from_file(audio_path)
        transcript = self.processor.parse_transcript(transcript_path)

        return spectrogram, transcript

    def __len__(self):
        return self.size
