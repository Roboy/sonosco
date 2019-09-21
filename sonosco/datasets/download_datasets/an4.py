import os
import click
import io
import shutil
import tarfile
import logging
import sonosco.common.audio_tools as audio_tools
import sonosco.common.path_utils as path_utils

from sonosco.datasets.download_datasets.create_manifest import create_manifest
from sonosco.common.utils import setup_logging
from sonosco.common.constants import *

LOGGER = logging.getLogger(__name__)

AN4_URL = 'http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'


def try_download_an4(target_dir, sample_rate, min_duration, max_duration):
    path_to_data = os.path.join(os.path.expanduser("~"), target_dir)
    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    target_unpacked_dir = os.path.join(path_to_data, "an4_unpacked")
    path_utils.try_create_directory(target_unpacked_dir)

    extracted_dir = os.path.join(path_to_data, "An4")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)
    LOGGER.info("Start downloading...")
    file_name = AN4_URL.split("/")[-1]

    target_filename = os.path.join(target_unpacked_dir, file_name)
    path_utils.try_download(target_filename, AN4_URL)
    LOGGER.info("Download complete")
    LOGGER.info("Unpacking...")
    tar = tarfile.open(target_filename)
    tar.extractall(extracted_dir)
    tar.close()
    assert os.path.exists(extracted_dir), f"Archive {file_name} was not properly uncompressed"
    LOGGER.info("Converting files to wav and extracting transcripts...")

    create_wav_and_transcripts(path_to_data, 'train', sample_rate, extracted_dir, 'an4_clstk')
    create_wav_and_transcripts(path_to_data, 'test', sample_rate, extracted_dir, 'an4test_clstk')

    create_manifest(path_to_data, os.path.join(path_to_data,'an4_train_manifest.csv'), min_duration, max_duration)
    create_manifest(path_to_data, os.path.join(path_to_data,'an4_val_manifest.csv'), min_duration, max_duration)


def create_wav_and_transcripts(path, data_tag, sample_rate, extracted_dir, wav_subfolder_name):
    tag_path = os.path.join(path,data_tag)
    transcript_path_new = os.path.join(tag_path, 'txt')
    wav_path_new = os.path.join(tag_path, 'wav')

    path_utils.try_create_directory(transcript_path_new)
    path_utils.try_create_directory(wav_path_new)

    wav_path_ext = os.path.join(extracted_dir, 'an4/wav')
    file_ids = os.path.join(extracted_dir, f'an4/etc/an4_{data_tag}.fileids')
    transcripts_ext = os.path.join(extracted_dir, f'an4/etc/an4_{data_tag}.transcription')
    path = os.path.join(wav_path_ext, wav_subfolder_name)
    convert_audio_to_wav(path, sample_rate)
    format_files(file_ids, transcript_path_new, wav_path_new, transcripts_ext, wav_path_ext)


def convert_audio_to_wav(train_path, sample_rate):
    with os.popen('find %s -type f -name "*.raw"' % train_path) as pipe:
        for line in pipe:
            raw_path = line.strip()
            new_path = line.replace('.raw', '.wav').strip()
            audio_tools.transcode_recordings_an4(raw_path=raw_path, wav_path= new_path, sample_rate=sample_rate)


def format_files(file_ids, new_transcript_path, new_wav_path, transcripts, wav_path):
    with open(file_ids, 'r') as f:
        with open(transcripts, 'r') as t:
            paths = f.readlines()
            transcripts = t.readlines()
            for x in range(len(paths)):
                path = os.path.join(wav_path, paths[x].strip()) + '.wav'
                filename = path.split('/')[-1]
                extracted_transcript = _process_transcript(transcripts, x)
                current_path = os.path.abspath(path)
                new_path = os.path.join(new_wav_path ,filename)
                text_path = os.path.join(new_transcript_path,filename.replace('.wav', '.txt'))
                with io.FileIO(text_path, "w") as file:
                    file.write(extracted_transcript.encode('utf-8'))
                os.rename(current_path, new_path)


def _process_transcript(transcripts, x):
    extracted_transcript = transcripts[x].split('(')[0].strip("<s>").split('<')[0].strip().upper()
    return extracted_transcript


@click.command()
@click.option("--target-dir", default="temp/data/an4", type=str, help="Directory to store the dataset.")
@click.option("--sample-rate", default=16000, type=int, help="Sample rate.")
@click.option("--min-duration", default=1, type=int,
              help="Prunes training samples shorter than the min duration (given in seconds).")
@click.option("--max-duration", default=15, type=int,
              help="Prunes training samples longer than the max duration (given in seconds).")
def main(**kwargs):
    """Processes and downloads an4 dataset."""
    try_download_an4(**kwargs)


if __name__ == '__main__':
    LOGGER = logging.getLogger(SONOSCO)
    setup_logging(LOGGER)
    main()
