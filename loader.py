import math
import warnings
from typing import Tuple

import librosa
import numpy as np
import torch
import torchaudio
from scipy import signal
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.distributed import get_rank
from torch.distributed import get_world_size

windows = {"bartlett": torch.bartlett_window,
           "blackman": torch.blackman_window,
           "hamming": torch.hamming_window,
           "hann": torch.hann_window}

windows_legacy = {'hamming': signal.hamming,
                  'hann': signal.hann,
                  'blackman': signal.blackman,
                  'bartlett': signal.bartlett}


class DataProcessor(object):
    def __init__(self, audio_conf, labels="abc", normalize=False, augment=False, legacy=True):
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
        self.window = windows_legacy.get(audio_conf["window"], windows_legacy["hamming"]) if legacy else windows.get(audio_conf["window"], windows["hamming"])
        self.normalize = normalize
        self.augment = augment
        self.legacy = legacy
        self.transform = torchaudio.transforms.Spectrogram(n_fft=int(self.sample_rate * self.window_size),
                                                           hop=int(self.sample_rate * self.window_stride),
                                                           window=self.window, normalize=self.normalize)

    @staticmethod
    def retrieve_file(audio_path, legacy=True):
        sound, sample_rate = torchaudio.load(audio_path)
        if legacy:
            sound = sound.numpy().T
            if len(sound.shape) > 1:
                if sound.shape[1] == 1:
                    sound = sound.squeeze()
                else:
                    sound = sound.mean(axis=1)
        return sound, sample_rate

    @staticmethod
    def augment_audio(sound, tempo_range: Tuple = (0.85, 1.15), gain_range: Tuple = (-6, 8)):
        """
        Changes tempo and gain of the wave
        """
        warnings.warn("Augmentation is not implemented")  # TODO: Implement
        return sound

    def parse_audio(self, audio_path):
        sound, sample_rate = self.retrieve_file(audio_path, self.legacy)
        if sample_rate != self.sample_rate:
            raise ValueError(f"The stated sample rate {self.sample_rate} and the factual rate {sample_rate} differ!")

        if self.augment:
            sound = self.augment_audio(sound)

        if self.legacy:
            n_fft = int(self.sample_rate * self.window_size)
            win_length = n_fft
            hop_length = int(self.sample_rate * self.window_stride)
            # STFT
            D = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=self.window)
            spectrogram, phase = librosa.magphase(D)
            # S = log(S+1)

            spectrogram = torch.FloatTensor(np.log1p(spectrogram))
        else:
            # TODO: Why these are different from librosa.stft?
            sound = sound.cuda()
            spectrogram = self.transform(sound)[-1, :, :].transpose(0, 1)

            # spectrogram = torch.stft(torch.from_numpy(sound.numpy().T.squeeze()),
            #                          n_fft=int(self.sample_rate * self.window_size),
            #                          hop_length=int(self.sample_rate * self.window_stride),
            #                          win_length=int(self.sample_rate * self.window_size),
            #                          window=torch.hamming_window(int(self.sample_rate * self.window_size)))[:, :, -1]

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
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False, legacy=True):
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
        self.processor = DataProcessor(audio_conf, labels, normalize, augment, legacy)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]

        spectrogram = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript_path)

        return spectrogram, transcript

    def __len__(self):
        return self.size


# TODO: Optimise
def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = batch[0][0]
    freq_size, max_seqlength = longest_sample.size()
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)

    # TODO: Numpy broadcasting magic
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)

        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)

        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    # TODO: Numpy broadcasting magic

    return inputs, targets, input_percentages, target_sizes


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        # TODO: Optimise
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


# TODO: Optimise
class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]
