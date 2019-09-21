#!/usr/bin/python3.7
import logging
import click
import torch
import sys

from collections import defaultdict
from sonosco.models.seq2seq_las import Seq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.decoders import GreedyDecoder
from sonosco.training.metrics.word_error_rate import word_error_rate
from sonosco.training.metrics.character_error_rate import character_error_rate
from sonosco.training.losses import cross_entropy_loss
from sonosco.config.global_settings import CUDA_ENABLED
from sonosco.serialization import Deserializer

LOGGER = logging.getLogger(SONOSCO)

EOS = '$'
SOS = '#'
PADDING_VALUE = '%'


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/train_seq2seq_las_yuriy.yaml",
              type=click.STRING, help="Path to train configurations.")
def main(config_path):
    config = parse_yaml(config_path)["train"]

    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    char_list = config["labels"] + EOS + SOS

    config["decoder"]["vocab_size"] = len(char_list)
    config["decoder"]["sos_id"] = char_list.index(SOS)
    config["decoder"]["eos_id"] = char_list.index(EOS)

    # Create mode
    if not config.get('checkpoint_path'):
        LOGGER.info("No checkpoint path specified")
        sys.exit(1)

    LOGGER.info(f"Loading model from checkpoint: {config['checkpoint_path']}")
    loader = Deserializer()
    model = loader.deserialize(Seq2Seq, config["checkpoint_path"])
    model.to(device)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(**config)

    # Create model trainer
    trainer = ModelTrainer(model, loss=cross_entropy_loss, epochs=config["max_epochs"],
                           train_data_loader=train_loader, val_data_loader=val_loader, test_data_loader=test_loader,
                           lr=config["learning_rate"], weight_decay=config['weight_decay'],
                           metrics=[word_error_rate, character_error_rate],
                           decoder=GreedyDecoder(config['labels']),
                           device=device, test_step=config["test_step"], custom_model_eval=True)

    metrics = defaultdict()
    trainer._compute_validation_error(metrics)
    LOGGER.info(metrics)


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
