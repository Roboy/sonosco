import logging
import click
import torch

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.models import TDSSeq2Seq
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.config.global_settings import CUDA_ENABLED


LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/infer.yaml", type=click.STRING,
              help="Path to infer configuration file.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
def main(config_path, audio_path):
    config = parse_yaml(config_path)["infer"]
    config['decoder']['vocab_dim'] = len(config['labels'])
    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    model = TDSSeq2Seq(config['labels'], config["encoder"], config["decoder"])
    model.to(device)
    model.eval()
    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    processor = AudioDataProcessor(**config)
    spect = processor.parse_audio(audio_path)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

    print(decoded_output)


if __name__ == "__main__":
    setup_logging(LOGGER)
    main()

