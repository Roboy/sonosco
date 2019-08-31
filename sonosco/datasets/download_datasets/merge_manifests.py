import os
import click
import io
import logging
import sonosco.common.path_utils as path_utils
from sonosco.datasets.download_datasets.create_manifest import order_and_prune_files
from sonosco.common.utils import setup_logging
import sonosco.common.audio_tools as audio_tools
from sonosco.common.constants import *
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--merge-dir", default="temp/data", type=str, help="Directory to the folder , all the manifest are stored in.")
@click.option("--min-duration", default=None, type=int, help="If provided, prunes any samples shorter than the min duration.")
@click.option("--max-duration", default=None, type=int, help="If provided, prunes any samples longer than the max duration")
@click.option("--output-path", default="temp/data/manifests/combined_manifest.csv", type=str, help="Output path, to store manifest.")

def main(merge_dir, min_duration, max_duration, output_path):
    global LOGGER
    LOGGER = logging.getLogger(SONOSCO)
    setup_logging(LOGGER)
    LOGGER.info("Start merging manifests...")

    merge_dir = os.path.join(os.path.expanduser("~"), merge_dir)
    output_path = os.path.join(os.path.expanduser("~"), output_path)
    output_directory = output_path[:-len(output_path.split("/")[-1])][:-1]
    path_utils.try_create_directory(output_directory)
    file_paths = []
    for dir_path, sub_dir, files in os.walk(merge_dir):
        for file in files:
            if file.endswith(".csv"):
                if dir_path == output_directory:
                    continue
                LOGGER.info(f"Found manifest: {file}")
                dir_path = os.path.join(merge_dir, dir_path)
                with open(os.path.join(dir_path, file), 'r') as fh:
                    file_paths += fh.readlines()
    file_paths = [file_path.split(',')[0] for file_path in file_paths]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    LOGGER.info("Creating final manifest")
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            duration = audio_tools.get_duration(wav_path)
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + ',' + duration + '\n'
            file.write(sample.encode('utf-8'))
    LOGGER.info("Final manifest was created!")

if __name__=="__main__":
    main()
