import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Sampler


class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        """Creates a data loader for AudioDatasets."""
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    # TODO: Optimise
    def _collate_fn(self, batch):
        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
        longest_sample = batch[0][0]
        freq_size, max_seqlength = longest_sample.size()
        minibatch_size = len(batch)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_percentages = torch.FloatTensor(minibatch_size)
        target_sizes = np.zeros(minibatch_size, dtype=np.int32)

        # TODO: Numpy broadcasting magic
        targets = []

        for x in range(minibatch_size):
            inputs[x][0].narrow(1, 0, batch[x][0].size(1)).copy_(batch[x][0])
            input_percentages[x] = batch[x][0].size(1) / float(max_seqlength)
            target_sizes[x] = len(batch[x][1])
            targets.extend(batch[x][1])

        return inputs, torch.IntTensor(targets), input_percentages, torch.from_numpy(target_sizes)