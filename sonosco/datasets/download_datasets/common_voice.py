import os
import wget
import click
import logging
import tarfile
import shutil
import csv
import sonosco.common.audio_tools as audio_tools
import sonosco.common.path_utils as path_utils
from multiprocessing.pool import ThreadPool
import subprocess
from sonosco.datasets.download_datasets.data_utils import create_manifest
from sonosco.common.utils import setup_logging

logger = logging.getLogger("sonosco")

COMMON_VOICE_URL = "https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz"

@click.command()
@click.option("--target-dir", default="temp/data/libri_speech", type=str, help="Directory to store the dataset.")
@click.option("--sample-rate", default=16000, type=int, help="Sample rate.")
@click.option("--files-to-use", multiple=True,
              default=["cv-valid-dev.csv","cv-valid-test.csv","cv-valid-train.csv"])
@click.option("--min-duration", default=1, type=int,
              help="Prunes training samples shorter than the min duration (given in seconds).")
@click.option("--max-duration", default=15, type=int,
              help="Prunes training samples longer than the max duration (given in seconds).")

def convert_to_wav(csv_file, target_dir, sample_rate):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to target_dir.
    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        target_dir: str, path to dir to save results; wav/ and txt/ dirs will be created
    """
    wav_dir = os.path.join(target_dir, 'wav/')
    txt_dir = os.path.join(target_dir, 'txt/')
    path_utils.try_create_directory(wav_dir)
    path_utils.try_create_directory(txt_dir)
    path_to_data = os.path.dirname(csv_file)

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.strip().upper()
        with open(os.path.join(txt_dir, file_name + '.txt'), 'w') as f:
            f.write(text)
        cmd = "sox {} -r {} -b 16 -c 1 {}".format(
            os.path.join(path_to_data, file_path),
            sample_rate,
            os.path.join(wav_dir, file_name + '.wav'))
        subprocess.call([cmd], shell=True)

    print('Converting mp3 to wav for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        data = [(row['filename'], row['text']) for row in reader]
        with ThreadPool(10) as pool:
            pool.map(process, data)


def main(target_dir, sample_rate, files_to_use, min_duration, max_duration):
    setup_logging(logger)

    path_to_data = os.path.join(os.path.expanduser("~"), target_dir)
    path_utils.try_create_directory(path_to_data)

    target_unpacked_dir = os.path.join(target_dir, "common_unpacked")
    path_utils.try_create_directory(target_unpacked_dir)

    extracted_dir = os.path.join(path_to_data, "CommonVoice")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)

    path_utils.try_download(target_unpacked_dir, COMMON_VOICE_URL)

    logger.info("Download complete")
    logger.info("Unpacking...")

    print("Unpacking corpus to {} ...".format(target_unpacked_dir))
    tar = tarfile.open(target_unpacked_dir)
    tar.extractall(extracted_dir)
    tar.close()

    for csv_file in args.files_to_process.split(','):
        convert_to_wav(os.path.join(extracted_dir, 'cv_corpus_v1/', csv_file),
                       os.path.join(target_dir, os.path.splitext(csv_file)[0]),
                       sample_rate)

    print('Creating manifests...')
    for csv_file in files_to_use.split(','):
        create_manifest(os.path.join(target_dir, os.path.splitext(csv_file)[0]),
                        os.path.splitext(csv_file)[0] + '_manifest.csv',
                        min_duration,
                        max_duration)


if __name__ == "__main__":
    main()