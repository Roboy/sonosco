import numpy as np
import logging
import torch

from torch.utils.data import Dataset, DataLoader, Sampler
from .audio_dataset import AudioDataProcessor, AudioDataset
from .data_sampler import BucketingSampler


LOGGER = logging.getLogger(__name__)


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
        input_lengths = torch.IntTensor(minibatch_size)
        target_sizes = np.zeros(minibatch_size, dtype=np.int32)

        # TODO: Numpy broadcasting magic
        targets = []

        for x in range(minibatch_size):
            inputs[x][0].narrow(1, 0, batch[x][0].size(1)).copy_(batch[x][0])
            input_lengths[x] = batch[x][0].size(1)
            target_sizes[x] = len(batch[x][1])
            targets.extend(batch[x][1])

        return inputs, torch.IntTensor(targets), input_lengths, torch.from_numpy(target_sizes)


def create_data_loaders(**kwargs):
    processor = AudioDataProcessor(**kwargs)

    # create train loader
    train_dataset = AudioDataset(processor, manifest_filepath=kwargs["train_manifest"])
    LOGGER.info(f"Training dataset containing {len(train_dataset)} samples is created")
    sampler = BucketingSampler(train_dataset, batch_size=kwargs["batch_size"])
    train_loader = AudioDataLoader(dataset=train_dataset, num_workers=kwargs["num_data_workers"], batch_sampler=sampler)
    LOGGER.info("Training data loader created.")

    # create validation loader
    val_dataset = AudioDataset(processor, manifest_filepath=kwargs["val_manifest"])
    LOGGER.info(f"Validation dataset containing {len(val_dataset)} samples is created")
    sampler = BucketingSampler(val_dataset, batch_size=kwargs["batch_size"])
    val_loader = AudioDataLoader(dataset=val_dataset, num_workers=kwargs["num_data_workers"], batch_sampler=sampler)
    LOGGER.info("Validation data loader created.")

    return train_loader, val_loader
