import logging
import click
import torch
import torch.nn.functional as torch_functional

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.datasets.processor import AudioDataProcessor

LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-m", "--model_path", default="pretrained/deepspeech_final.pth", type=click.STRING,
              help="Path to a pretrained model.")
@click.option("-d", "--decoder", default="greedy", type=click.STRING, help="Type of decoder.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
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
def main(model_path, cuda, audio_path, **kwargs):
    device = torch.device("cuda" if cuda else "cpu")
    model = DeepSpeech2.load_model(model_path)
    model.eval()
    decoder = BeamCTCDecoder(model.labels, blank_index=model.labels.index('_'))
    processor = AudioDataProcessor(**model.audio_conf)

    spect = processor.parse_audio(audio_path)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    print(decoded_output)


if __name__ == "__main__":
    main()

