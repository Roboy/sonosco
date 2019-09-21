import logging
import click
import torch

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.models import TDSSeq2Seq
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.common.global_settings import CUDA_ENABLED
from sonosco.serialization import Deserializer


LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/infer.yaml", type=click.STRING,
              help="Path to infer configuration file.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
@click.option("-p", "--plot", default=False, is_flag=True, help="Show plots.")
def main(config_path, audio_path, plot):
    config = parse_yaml(config_path)["infer"]
    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    loader = Deserializer()
    model = loader.deserialize(TDSSeq2Seq, config["model_checkpoint_path"])
    model.to(device)
    model.eval()

    decoder = GreedyDecoder(model.decoder.labels)

    processor = AudioDataProcessor(**config)
    spect, lens = processor.parse_audio_for_inference(audio_path)
    spect = spect.to(device)

    # Watch out lens is modified after this call!
    # It is now equal to the number of encoded states
    with torch.no_grad():
        out, output_lens, attention = model(spect, lens)
        decoded_output, decoded_offsets = decoder.decode(out, output_lens)
        LOGGER.info(decoded_output)
        if plot:
            import matplotlib.pyplot as plt
            plt.matshow(attention[0].numpy())
            plt.show()


if __name__ == "__main__":
    setup_logging(LOGGER)
    main()

