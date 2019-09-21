import logging
import click
import torch

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.models import Seq2Seq
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.config.global_settings import DEVICE
from sonosco.serialization import Deserializer


LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/infer_las.yaml", type=click.STRING,
              help="Path to infer configuration file.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
def main(config_path, audio_path):
    config = parse_yaml(config_path)["infer"]

    loader = Deserializer()
    model: Seq2Seq = loader.deserialize(Seq2Seq, config["model_checkpoint_path"])
    model.to(DEVICE)
    model.eval()

    decoder = GreedyDecoder(config["labels"])

    processor = AudioDataProcessor(**config)
    spect, lens = processor.parse_audio_for_inference(audio_path)
    spect = spect.to(DEVICE)

    with torch.no_grad():
        output = model.recognize(spect[0], lens, config["labels"], config["recognizer"])[0]
        transcription = decoder.convert_to_strings(torch.tensor([output['yseq']]))
        LOGGER.info(transcription)


if __name__ == "__main__":
    setup_logging(LOGGER)
    main()

