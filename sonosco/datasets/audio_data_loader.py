import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Sampler


class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        '''
        Creates a data loader for AudioDatasets.
        '''
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        #sort the batch in decreasing order of sequence length
        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)

        #pad the tensors to have equal lengths, therefore transpose the tensors in
        #the batch. The tensors have shape: freq_size x sequence_length
        #and need to be of shape: sequence_length x  freq_length, as sequence length differs
        #but not the freq_length
        inputs = torch.nn.utils.rnn.pad_sequence(list(map(lambda x: x[0].transpose(0,1), batch)), batch_first=True)

        #inputs need to be transposed back from shape batch_size x sequence_length x  freq_length
        #to batch_size x freq_length x sequence_length. Additionally, unsqueeze tensor
        inputs = inputs.transpose(1,2).unsqueeze(1)
        input_lengths = torch.IntTensor(list(map(lambda x: x[0].size(1), batch))) #create tensor of input lengths

        targets_arr = list(zip(*batch))[1] #extract targets array from batch ( batch is array of tuples)
        target_lengths = torch.IntTensor(list(map(lambda x: len(x),targets_arr))) #create tensor of target lengths
        targets = torch.cat(list(map(lambda x: torch.IntTensor(x), targets_arr))) #create tensor of targets

        return inputs, targets, input_lengths, target_lengths
