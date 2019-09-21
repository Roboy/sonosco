import os
import click
import tarfile
import shutil
import logging
import sonosco.common.audio_tools as audio_tools
import sonosco.common.path_utils as path_utils

from sonosco.datasets.download_datasets.create_manifest import create_manifest
from sonosco.common.utils import setup_logging
from sonosco.common.constants import *
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

LIBRI_SPEECH_URLS = {
    #"train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
              #          "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
              #          "http://www.openslr.org/resources/12/train-other-500.tar.gz"
    #          ],

    #"val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
    #        "http://www.openslr.org/resources/12/dev-other.tar.gz"
    #       ],

    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"]  #,
    #"test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


def try_download_librispeech(target_dir, sample_rate, files_to_use, min_duration, max_duration):
    path_to_data = os.path.join(os.path.expanduser("~"), target_dir)
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)

    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(path_to_data, split_type)
        path_utils.try_create_directory(split_dir)
        split_wav_dir = os.path.join(split_dir, "wav")
        path_utils.try_create_directory(split_wav_dir)
        split_txt_dir = os.path.join(split_dir, "txt")
        path_utils.try_create_directory(split_txt_dir)
        extracted_dir = os.path.join(split_dir, "LibriSpeech")

        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)

        for url in lst_libri_urls:
            # check if we want to dl this file
            dl_flag = False
            for f in files_to_use:
                if url.find(f) != -1:
                    dl_flag = True
            if not dl_flag:
                LOGGER.info(f"Skipping url: {url}")
                continue

            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)
            LOGGER.info(f"Downloading from {url}")
            path_utils.try_download(target_filename, url)
            LOGGER.info("Download complete")
            LOGGER.info(f"Unpacking {filename}...")
            tar = tarfile.open(target_filename)
            tar.extractall(split_dir)
            tar.close()
            os.remove(target_filename)
            assert os.path.exists(extracted_dir), f"Archive {filename} was not properly uncompressed"

            LOGGER.info("Converting flac files to wav and extracting transcripts...")
            for root, subdirs, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.find(".flac") != -1:
                        _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir,
                                      base_filename=f, root_dir=root, sample_rate=sample_rate)

            LOGGER.info(f"Finished {url}")
            shutil.rmtree(extracted_dir)

        manifest_path = os.path.join(path_to_data, f"libri_{split_type}_manifest.csv")
        if os.path.exists(manifest_path):
            continue

        if split_type == 'train':  # Prune to min/max duration
            create_manifest(split_dir, manifest_path, min_duration, max_duration)
        else:
            create_manifest(split_dir, manifest_path)


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def _process_file(wav_dir, txt_dir, base_filename, root_dir, sample_rate):
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    audio_tools.transcode_recording(full_recording_path, wav_recording_path, sample_rate)
    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), f"Transcript file {transcript_file} does not exist"
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, f"{key} is not in the transcriptions"
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


@click.command()
@click.option("--target-dir", default="temp/data/libri_speech", type=str, help="Directory to store the dataset.")
@click.option("--sample-rate", default=16000, type=int, help="Sample rate.")
@click.option("--files-to-use", multiple=True,
              default=["train-clean-100.tar.gz", "train-clean-360.tar.gz", "train-other-500.tar.gz",
                       "dev-clean.tar.gz", "dev-other.tar.gz", "test-clean.tar.gz", "test-other.tar.gz"],
              type=str, help="List of file names to download.")
@click.option("--min-duration", default=1, type=int,
              help="Prunes training samples shorter than the min duration (given in seconds).")
@click.option("--max-duration", default=15, type=int,
              help="Prunes training samples longer than the max duration (given in seconds).")

def main(**kwargs):
    """Processes and downloads LibriSpeech dataset."""
    try_download_librispeech(**kwargs)


if __name__ == "__main__":
    LOGGER = logging.getLogger(SONOSCO)
    setup_logging(LOGGER)
    main()
