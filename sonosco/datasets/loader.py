import logging
import torch
import torch.nn

from torch.utils.data import DataLoader
from .dataset import AudioDataProcessor, AudioDataset
from .samplers import BucketingSampler


LOGGER = logging.getLogger(__name__)


class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        '''
        Creates a data loader for AudioDatasets.
        '''
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        # sort the batch in decreasing order of sequence length
        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)

        # pad the tensors to have equal lengths, therefore transpose the tensors in
        # the batch. The tensors have shape: freq_size x sequence_length
        # and need to be of shape: sequence_length x  freq_length, as sequence length differs
        # but not the freq_length
        inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0].transpose(0, 1), batch)), batch_first=True,
                                                 padding_value=-1)

        # inputs need to be transposed back from shape batch_size x sequence_length x  freq_length
        # to batch_size x freq_length x sequence_length. Additionally, unsqueeze tensor
        inputs = inputs.transpose(1, 2).unsqueeze(1)
        input_lengths = torch.IntTensor(list(map(lambda x: x[0].size(1), batch))) # create tensor of input lengths

        targets_arr = list(zip(*batch))[1] # extract targets array from batch ( batch is array of tuples)
        target_lengths = torch.IntTensor(list(map(lambda x: len(x), targets_arr))) # create tensor of target lengths
        targets = torch.cat(list(map(lambda x: torch.IntTensor(x), targets_arr))) # create tensor of targets

        return inputs, targets, input_lengths, target_lengths


def create_data_loaders(**kwargs):
    processor = AudioDataProcessor(**kwargs)

    # create train loader
    train_dataset = AudioDataset(processor, manifest_filepath=kwargs["train_manifest"])
    LOGGER.info(f"Training dataset containing {len(train_dataset)} samples is created")
    sampler = BucketingSampler(train_dataset, batch_size=kwargs["batch_size"])
    train_loader = AudioDataLoader(dataset=train_dataset, num_workers=kwargs["num_data_workers"], batch_sampler=sampler,
                                   pin_memory=True)
    LOGGER.info("Training data loader created.")

    # create validation loader
    val_dataset = AudioDataset(processor, manifest_filepath=kwargs["val_manifest"])
    LOGGER.info(f"Validation dataset containing {len(val_dataset)} samples is created")
    sampler = BucketingSampler(val_dataset, batch_size=kwargs["batch_size"])
    val_loader = AudioDataLoader(dataset=val_dataset, num_workers=kwargs["num_data_workers"], batch_sampler=sampler,
                                 pin_memory=True)
    LOGGER.info("Validation data loader created.")

    # create validation loader
    test_dataset = AudioDataset(processor, manifest_filepath=kwargs["test_manifest"])
    LOGGER.info(f"Test dataset containing {len(test_dataset)} samples is created")
    sampler = BucketingSampler(test_dataset, batch_size=kwargs["batch_size"])
    test_loader = AudioDataLoader(dataset=test_dataset, num_workers=kwargs["num_data_workers"], batch_sampler=sampler,
                                  pin_memory=True)
    LOGGER.info("Test data loader created.")

    return train_loader, val_loader, test_loader
