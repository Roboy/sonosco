import logging
import click
import torch
import torch.nn.functional as torch_functional

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.datasets import create_data_loaders
from sonosco.models import DeepSpeech2, TDSSeq2Seq
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.config.global_settings import CUDA_ENABLED

LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/infer.yaml", type=click.STRING,
              help="Path to infer configuration file.")
@click.option("-a", "--audio_path", default="audio.wav", type=click.STRING, help="Path to an audio file.")
def main(config_path, audio_path):
    config = parse_yaml(config_path)["train"]
    config['decoder']['vocab_dim'] = len(config['labels'])
    train_loader, val_loader = create_data_loaders(**config)

    def cross_entropy_loss(batch, model):
        batch_x, batch_y, input_lengths, target_lengths = batch
        # check out the _collate_fn in loader to understand the next transformations
        batch_x = batch_x.squeeze(1).transpose(1, 2)
        batch_y = torch.split(batch_y, target_lengths.tolist())
        model_output, lens, loss = model(batch_x, input_lengths, batch_y)
        return loss, (model_output, lens)

    device = torch.device("cuda" if CUDA_ENABLED else "cpu")
    model = TDSSeq2Seq(config['labels'], config["encoder"], config["decoder"])
    model.to(device)
    model.eval()
    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
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

