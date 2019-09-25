import numpy as np

from torch.utils.data import Sampler


class BucketingSampler(Sampler):
    def __init__(self, data_source: any, batch_size: int=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        Args:
            data_source: source of data
            batch_size: size of batch
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        # TODO: Optimise
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        self.shuffle()
        for ids in self.bins:
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        """
        Shuffles the buckets

        """
        np.random.shuffle(self.bins)
