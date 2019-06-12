import logging
import os

from sonosco.common.constants import SONOSCO
from sonosco.datasets.audio_dataset import AudioDataset, DataProcessor
from sonosco.datasets.audio_data_sampler import BucketingSampler
from sonosco.datasets.audio_data_loader import DataLoader
from sonosco.datasets.download_datasets.librispeech import try_download_librispeech


LOGGER = logging.getLogger(SONOSCO)
LIBRI_SPEECH_DIR = "temp/test_data/libri_speech"


def test_librispeech_download():
    # prepare
    if os.path.exists(LIBRI_SPEECH_DIR):
        os.removedirs(LIBRI_SPEECH_DIR)

    # get manifest file
    manifest_directory = os.path.join(os.path.expanduser("~"), LIBRI_SPEECH_DIR)
    test_manifest = os.path.join(manifest_directory, "libri_test_clean_manifest.csv")

    if not os.path.exists(test_manifest):
        try_download_librispeech(LIBRI_SPEECH_DIR, 16000, ["test-clean.tar.gz", "test-other.tar.gz"], 1, 15)

    assert os.path.exists(test_manifest)


def test_librispeech_clean():
    # create data processor
    audio_conf = dict(sample_rate=16000, window_size=.02, window_stride=.01,
                      labels='ABCDEFGHIJKLMNOPQRSTUVWXYZ', normalize=True, augment=False)
    processor = DataProcessor(**audio_conf)

    # get manifest file
    manifest_directory = os.path.join(os.path.expanduser("~"), LIBRI_SPEECH_DIR)
    test_manifest = os.path.join(manifest_directory, "libri_test_clean_manifest.csv")

    if not os.path.exists(test_manifest):
        try_download_librispeech(LIBRI_SPEECH_DIR, 16000, ["test-clean.tar.gz", "test-other.tar.gz"], 1, 15)

    assert os.path.exists(test_manifest)

    # create audio dataset
    test_dataset = AudioDataset(processor, manifest_filepath=test_manifest)
    LOGGER.info("Dataset is created")
    test = test_dataset[0]
    batch_size = 16
    sampler = BucketingSampler(test_dataset, batch_size=batch_size)
    dataloader = DataLoader(dataset=test_dataset, num_workers=4, batch_sampler=sampler)
    test_dataset[0]

