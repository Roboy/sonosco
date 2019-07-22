import logging
import click
import torch
import os

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import try_create_directory
from sonosco.datasets.download_datasets.data_utils import create_manifest_wav_only
from sonosco.datasets import AudioDatasetWavOnly, BucketingSampler, AudioDataLoaderWavOnly
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor

LOGGER = logging.getLogger(SONOSCO)
MANIFEST = "manifest.txt"


def create_results(transcripts, filenames, output_path):
    try_create_directory(output_path)
    # transcripts dimensions: batch x beams x chars
    for transcript, filename in zip(transcripts, filenames):
        filename = os.path.basename(filename)
        filename, _ = os.path.splitext(filename)
        filename = f"{filename}.txt"

        with open(os.path.join(output_path, filename), "w") as file:
            # take the first most probable beam and write it into the result file
            file.write(transcript[0])


@click.command()
@click.option("-m", "--model_path", default="pretrained/deepspeech_final.pth", type=click.STRING,
              help="Path to a pretrained model.")
@click.option("-d", "--decoder", default="greedy", type=click.STRING, help="Type of decoder.")
@click.option("-a", "--audio_path", default="input", type=click.STRING, help="Path to audio files folder.")
@click.option("--cuda", is_flag=True, help="Should cuda be used.")
@click.option('--top-paths', default=1, type=click.INT, help='number of beams to return')
@click.option('--beam-width', default=10, type=click.INT, help='Beam width to use')
@click.option('--lm-path', default=None, type=click.STRING,
              help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
@click.option('--alpha', default=0.8, type=click.FLOAT, help='Language model weight')
@click.option('--beta', default=1, type=click.FLOAT, help='Language model word bonus (all words)')
@click.option('--cutoff-top-n', default=40, type=click.INT,
              help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                   'vocabulary will be used in beam search, default 40.')
@click.option('--cutoff-prob', default=1.0, type=click.FLOAT,
              help='Cutoff probability in pruning, default 1.0, no pruning.')
@click.option('--lm-workers', default=1, type=click.INT, help='Number of LM processes to use')
@click.option('--data_workers', default=5, type=click.INT, help='Number of data loading workers')
@click.option('--batch_size', default=16, type=click.INT, help='Batch size')
@click.option('--output_path', default="output", type=click.STRING, help='Output path')
def main(model_path, cuda, audio_path, **kwargs):
    create_manifest_wav_only(audio_path, "manifest.txt")

    device = torch.device("cuda" if cuda else "cpu")

    # Load model
    model = DeepSpeech2.load_model(model_path)
    model.eval()

    # Load decoder
    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    # Load data processor
    processor = AudioDataProcessor(**model.audio_conf)

    # Create data loader for the input audio files
    dataset = AudioDatasetWavOnly(processor, manifest_filepath=MANIFEST)
    LOGGER.info(f"Dataset containing {len(dataset)} samples is created")
    sampler = BucketingSampler(dataset, batch_size=kwargs["batch_size"])
    loader = AudioDataLoaderWavOnly(dataset=dataset, num_workers=kwargs["data_workers"], batch_sampler=sampler)
    LOGGER.info("Data loader created.")

    for inputs, input_lengths, filenames in loader:
        out, output_sizes = model(inputs, input_lengths)
        decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
        LOGGER.info(f"Transcribed {len(decoded_output)} files.")
        create_results(decoded_output, filenames, kwargs["output_path"])
        LOGGER.info(f"Results written out into the {kwargs['output_path']} directory.")


if __name__ == "__main__":
    setup_logging(LOGGER, filename="transcription")
    main()
