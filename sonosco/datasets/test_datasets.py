from AudioDataLoader import AudioDataLoader
from AudioDataSampler import BucketingSampler, DistributedBucketingSampler
from AudioDataset import AudioDataset


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
    dataloader = AudioDataLoader(test_dataset,num_workers=4, batch_sampler=sampler)

    inputs, targets, input_percentages, target_sizes = next(iter(dataloader))
    print(targets)
if __name__ == "__main__":
    main()