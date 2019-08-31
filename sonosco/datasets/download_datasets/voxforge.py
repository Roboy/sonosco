import os
import click
import logging
from six.moves import urllib
import argparse
import re
import tempfile
import shutil
import subprocess
import tarfile
import io
import sonosco.common.audio_tools as audio_tools
import sonosco.common.path_utils as path_utils
from sonosco.datasets.download_datasets.create_manifest import create_manifest
from sonosco.common.utils import setup_logging
from sonosco.common.constants import *
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

VOXFORGE_URL_16kHz = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'

def try_download_voxforge(target_dir, sample_rate, min_duration, max_duration):
    path_to_data = os.path.join(os.path.expanduser("~"), target_dir)
    path_utils.try_create_directory(path_to_data)

    LOGGER.info(f"Start downloading Voxforge from {VOXFORGE_URL_16kHz}")
    request = urllib.request.Request(VOXFORGE_URL_16kHz)
    response = urllib.request.urlopen(request)
    content = response.read()
    all_files = re.findall("href\=\"(.*\.tgz)\"", content.decode("utf-8"))
    for f in tqdm(all_files, total=len(all_files)):
        prepare_sample(f.replace(".tgz", ""), VOXFORGE_URL_16kHz + f, path_to_data, sample_rate)
    create_manifest(path_to_data, os.path.join(path_to_data,'voxforge_train_manifest.csv'), min_duration, max_duration)

def _get_recordings_dir(sample_dir, recording_name):
    wav_dir = os.path.join(sample_dir, recording_name, "wav")
    if os.path.exists(wav_dir):
        return "wav", wav_dir
    flac_dir = os.path.join(sample_dir, recording_name, "flac")
    if os.path.exists(flac_dir):
        return "flac", flac_dir
    raise Exception("wav or flac directory was not found for recording name: {}".format(recording_name))


def prepare_sample(recording_name, url, target_folder, sample_rate):
    """
    Downloads and extracts a sample from VoxForge and puts the wav and txt files into :target_folder.
    """
    wav_dir = os.path.join(target_folder, "wav")
    path_utils.try_create_directory(wav_dir)
    txt_dir = os.path.join(target_folder, "txt")
    path_utils.try_create_directory(txt_dir)
    # check if sample is processed
    filename_set = set(['_'.join(wav_file.split('_')[:-1]) for wav_file in os.listdir(wav_dir)])
    if recording_name in filename_set:
        return

    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    content = response.read()
    response.close()
    with tempfile.NamedTemporaryFile(suffix=".tgz", mode='wb') as target_tgz:
        target_tgz.write(content)
        target_tgz.flush()
        dirpath = tempfile.mkdtemp()

        tar = tarfile.open(target_tgz.name)
        tar.extractall(dirpath)
        tar.close()

        recordings_type, recordings_dir = _get_recordings_dir(dirpath, recording_name)
        tgz_prompt_file = os.path.join(dirpath, recording_name, "etc", "PROMPTS")

        if os.path.exists(recordings_dir) and os.path.exists(tgz_prompt_file):
            transcriptions = open(tgz_prompt_file).read().strip().split("\n")
            transcriptions = {t.split()[0]: " ".join(t.split()[1:]) for t in transcriptions}
            for wav_file in os.listdir(recordings_dir):
                recording_id = wav_file.split('.{}'.format(recordings_type))[0]
                transcription_key = recording_name + "/mfc/" + recording_id
                if transcription_key not in transcriptions:
                    continue
                utterance = transcriptions[transcription_key]

                target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(recording_name, recording_id))
                target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(recording_name, recording_id))
                with io.FileIO(target_txt_file, "w") as file:
                    file.write(utterance.encode('utf-8'))
                original_wav_file = os.path.join(recordings_dir, wav_file)
                audio_tools.transcode_recording(original_wav_file, target_wav_file, sample_rate)

        shutil.rmtree(dirpath)

@click.command()
@click.option("--target-dir", default="temp/data/voxforge", type=str, help="Directory to store the dataset.")
@click.option("--sample-rate", default=16000, type=int, help="Sample rate.")

@click.option("--min-duration", default=1, type=int,
              help="Prunes training samples shorter than the min duration (given in seconds).")
@click.option("--max-duration", default=15, type=int,
              help="Prunes training samples longer than the max duration (given in seconds).")



def main(**kwargs):
    global LOGGER
    LOGGER = logging.getLogger(SONOSCO)
    setup_logging(LOGGER)
    try_download_voxforge(**kwargs)


if __name__ == '__main__':
    main()