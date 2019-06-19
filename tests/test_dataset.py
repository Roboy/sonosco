import logging
import os
import pytest
import numpy as np
import librosa

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.datasets.audio_dataset import AudioDataset, AudioDataProcessor
from sonosco.datasets.data_sampler import BucketingSampler
from sonosco.datasets.data_loader import DataLoader
from sonosco.datasets.download_datasets.librispeech import try_download_librispeech


LIBRI_SPEECH_DIR = "temp/test_data/libri_speech"
TEST_WAVS_DIR = "test_wavs"
SAMPLE_RATE = 16000


@pytest.fixture
def logger():
    logger = logging.getLogger(SONOSCO)
    setup_logging(logger)
    return logger


def test_librispeech_download(logger):
    # prepare
    if os.path.exists(LIBRI_SPEECH_DIR):
        os.removedirs(LIBRI_SPEECH_DIR)

    # get manifest file
    manifest_directory = os.path.join(os.path.expanduser("~"), LIBRI_SPEECH_DIR)
    test_manifest = os.path.join(manifest_directory, "libri_test_clean_manifest.csv")

    if not os.path.exists(test_manifest):
        logger.info("Starting to download dataset")
        try_download_librispeech(LIBRI_SPEECH_DIR, 16000, ["test-clean.tar.gz", "test-other.tar.gz"], 1, 15)

    assert os.path.exists(test_manifest)


def test_librispeech_clean(logger):
    # create data processor
    audio_conf = dict(sample_rate=SAMPLE_RATE, window_size=.02, window_stride=.01,
                      labels='ABCDEFGHIJKLMNOPQRSTUVWXYZ', normalize=True, augment=True)
    processor = AudioDataProcessor(**audio_conf)

    # get manifest file
    manifest_directory = os.path.join(os.path.expanduser("~"), LIBRI_SPEECH_DIR)
    test_manifest = os.path.join(manifest_directory, "libri_test_clean_manifest.csv")

    if not os.path.exists(test_manifest):
        try_download_librispeech(LIBRI_SPEECH_DIR, SAMPLE_RATE, ["test-clean.tar.gz", "test-other.tar.gz"], 1, 15)

    assert os.path.exists(test_manifest)

    # create audio dataset
    test_dataset = AudioDataset(processor, manifest_filepath=test_manifest)
    logger.info("Dataset is created")

    if os.path.exists(TEST_WAVS_DIR):
        os.removedirs(TEST_WAVS_DIR)

    os.makedirs(TEST_WAVS_DIR)

    n_samples = len(test_dataset)

    ids = np.random.randint(n_samples, size=min(10, n_samples))

    for index in ids:
        sound, transcription = test_dataset.get_raw(index)
        librosa.output.write_wav(os.path.join(TEST_WAVS_DIR, f"audio_{index}.wav"), sound, SAMPLE_RATE)

    # batch_size = 16
    # sampler = BucketingSampler(test_dataset, batch_size=batch_size)
    # dataloader = DataLoader(dataset=test_dataset, num_workers=4, batch_sampler=sampler)
    # test_dataset[0]

