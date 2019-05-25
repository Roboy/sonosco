# ----------------------------------------------------------------------------
# Based on SeanNaren's deepspeech.pytorch:
# https://github.com/SeanNaren/deepspeech.pytorch
# ----------------------------------------------------------------------------

import warnings
from typing import Tuple

import torch
import torchaudio
from scipy import signal
from torch.utils.data import Dataset

windows = {"bartlett": torch.bartlett_window,
           "blackman": torch.blackman_window,
           "hamming": torch.hamming_window,
           "hann": torch.hann_window}


class DataProcessor(object):
    def __init__(self, audio_conf, labels="abc", normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.window_stride = audio_conf["window_stride"]
        self.window_size = audio_conf["window_size"]
        self.sample_rate = audio_conf["sample_rate"]
        self.window = windows.get(audio_conf["window"], windows["hamming"])
        self.normalize = normalize
        self.augment = augment

    @staticmethod
    def retrieve_file(audio_path):
        sound, sample_rate = torchaudio.load(audio_path)
        return sound, sample_rate

    @staticmethod
    def augment_audio(sound, tempo_range: Tuple = (0.85, 1.15), gain_range: Tuple = (-6, 8)):
        """
        Changes tempo and gain of the wave
        """
        warnings.warn("Augmentation is not implemented")  # TODO: Implement
        return sound

    def parse_audio(self, audio_path):
        sound, sample_rate = self.retrieve_file(audio_path)
        if sample_rate != self.sample_rate:
            raise ValueError(f"The stated sample rate {self.sample_rate} and the factual rate {sample_rate} differ!")

        if self.augment:
            sound = self.augment_audio(sound)

        #sound = sound.cuda()
        spectrogram = torch.stft(torch.from_numpy(sound.numpy().T.squeeze()),
                                  n_fft=int(self.sample_rate * self.window_size),
                                  hop_length=int(self.sample_rate * self.window_stride),
                                  win_length=int(self.sample_rate * self.window_size),
                                  window=torch.hamming_window(int(self.sample_rate * self.window_size)))[:, :, -1]



        if self.normalize:
            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram.add_(-mean)
            spectrogram.div_(std)

        return spectrogram

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        # TODO: Is it fast enough?
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript


class AudioDataset(Dataset):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(AudioDataset, self).__init__()
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.processor = DataProcessor(audio_conf, labels, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        spectrogram = self.processor.parse_audio(audio_path)
        transcript = self.processor.parse_transcript(transcript_path)

        return spectrogram, transcript

    def __len__(self):
        return self.size

def main():
    audio_conf = dict(sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming')
    test_manifest = '/Users/florianlay/data/libri_test_clean_manifest.csv'
    labels = 'abc'
    test_dataset = AudioDataset(audio_conf=audio_conf, manifest_filepath=test_manifest, labels=labels,
                                 normalize=False, augment=False)
    print("Dataset is created\n====================\n")

    test = test_dataset[0]
    batch_size = 16
    sampler = BucketingSampler(test_dataset, batch_size=batch_size)
    dataloader = DataLoader(dataset=test_dataset, num_workers=4, collate_fn=_collate_fn, batch_sampler=sampler)
    test_dataset[0]
    #inputs, targets, input_percentages, target_sizes = next(iter(dataloader))
if __name__ == "__main__":
    main()