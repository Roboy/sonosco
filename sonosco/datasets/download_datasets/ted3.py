import os
import click
import logging
import argparse
import subprocess
import unicodedata
import tarfile
import io
import shutil
import sonosco.common.audio_tools as audio_tools
import sonosco.common.path_utils as path_utils
from sonosco.datasets.download_datasets.create_manifest import create_manifest
from sonosco.common.utils import setup_logging
from sonosco.common.constants import *
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"

def try_download_ted3(target_dir, sample_rate, min_duration, max_duration):
    path_to_data = os.path.join(os.path.expanduser("~"), target_dir)
    path_utils.try_create_directory(path_to_data)

    target_unpacked_dir = os.path.join(path_to_data, "ted3_unpacked")
    path_utils.try_create_directory(target_unpacked_dir)

    extracted_dir = os.path.join(path_to_data, "Ted3")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    LOGGER.info(f"Start downloading Ted 3 from {TED_LIUM_V2_DL_URL}")
    file_name = TED_LIUM_V2_DL_URL.split("/")[-1]
    target_filename = os.path.join(target_unpacked_dir, file_name)
    path_utils.try_download(target_filename, TED_LIUM_V2_DL_URL)

    LOGGER.info("Download complete")
    LOGGER.info("Unpacking...")
    tar = tarfile.open(target_filename)
    tar.extractall(extracted_dir)
    tar.close()
    os.remove(target_unpacked_dir)
    assert os.path.exists(extracted_dir), f"Archive {file_name} was not properly uncompressed"
    LOGGER.info("Converting files to wav and extracting transcripts...")
    prepare_dir(path_to_data, sample_rate)
    create_manifest(path_to_data, os.path.join(path_to_data,'ted3_train_manifest.csv'), min_duration, max_duration)


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec


def prepare_dir(ted_dir, sample_rate):
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(ted_dir, "wav")
    path_utils.try_create_directory(wav_dir)
    txt_dir = os.path.join(ted_dir, "txt")
    path_utils.try_create_directory(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
        stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
        all_utterances = get_utterances_from_stm(stm_file_full)

        all_utterances = filter(filter_short_utterances, all_utterances)
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
            target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
            audio_tools.transcode_recordings_ted3(sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                          sample_rate=sample_rate)
            with io.FileIO(target_txt_file, "w") as f:
                f.write(_preprocess_transcript(utterance["transcript"]).encode('utf-8'))
        counter += 1

@click.command()
@click.option("--target-dir", default="temp/data/ted3", type=str, help="Directory to store the dataset.")
@click.option("--sample-rate", default=16000, type=int, help="Sample rate.")

@click.option("--min-duration", default=1, type=int,
              help="Prunes training samples shorter than the min duration (given in seconds).")
@click.option("--max-duration", default=15, type=int,
              help="Prunes training samples longer than the max duration (given in seconds).")


def main(**kwargs):
    global LOGGER
    LOGGER = logging.getLogger(SONOSCO)
    setup_logging(LOGGER)
    try_download_ted3(**kwargs)

if __name__ == "__main__":
    main()
