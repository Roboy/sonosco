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
from sonosco.model.deserializer import ModelDeserializer


LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/infer.yaml", type=click.STRING,
              help="Path to infer configuration file.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
def main(config_path, audio_path):
    config = parse_yaml(config_path)["infer"]
    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    loader = ModelDeserializer()
    model = loader.deserialize_model(TDSSeq2Seq, config["model_checkpoint_path"])
    model.to(device)
    model.eval()

    decoder = GreedyDecoder(model.decoder.labels)

    processor = AudioDataProcessor(**config)
    spect, lens = processor.parse_audio_for_inference(audio_path)
    spect = spect.to(device)

    out, output_sizes = model(spect, lens)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)

    print(decoded_output)


if __name__ == "__main__":
    setup_logging(LOGGER)
    main()

