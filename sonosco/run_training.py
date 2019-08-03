import logging
import click
import torch.nn.functional as torch_functional
import json

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.training.word_error_rate import WER
from sonosco.training.character_error_rate import CER
from sonosco.training.tensorboard_callback import TensorBoardCallback

LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-e", "--experiment_name", default="default", type=click.STRING, help="Experiment name.")
@click.option("-c", "--config_path", default="config/train.yaml", type=click.STRING,
              help="Path to train configurations.")
@click.option("-l", "--log_dir", default="log/", type=click.STRING, help="Log directory for tensorboard.")

def main(experiment_name, config_path, log_dir):
    Experiment.create(experiment_name)
    config = parse_yaml(config_path)["train"]
    with open(config["labels_path"]) as label_file:
        labels = str(''.join(json.load(label_file)))

    train_loader, val_loader = create_data_loaders(**config, labels=labels)

    def custom_loss(batch, model):
        batch_x, batch_y, input_lengths, target_lengths = batch
        model_output, output_lengths = model(batch_x, input_lengths)
        loss = torch_functional.ctc_loss(model_output.transpose(0, 1), batch_y, output_lengths, target_lengths)
        return loss, (model_output, output_lengths)

    # TODO: change to load different models dynamically
    model = DeepSpeech2(labels=labels)

    if config["decoder"] == GreedyDecoder.__name__:
        decoder = GreedyDecoder(labels=labels)
    elif config["decoder"]==BeamCTCDecoder.__name__:
        decoder = BeamCTCDecoder(labels=labels)
    trainer = ModelTrainer(model, loss=custom_loss, epochs=config["max_epochs"],
                           train_data_loader=train_loader, val_data_loader=val_loader,
                           lr=config["learning_rate"], custom_model_eval=True,
                           decoder= decoder, metrics=[WER, CER], callbacks=[TensorBoardCallback(log_dir)])

    try:
        trainer.start_training()
    except KeyboardInterrupt:
        trainer.stop_training()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
