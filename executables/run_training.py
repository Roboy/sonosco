import logging
import click
import torch.nn.functional as torch_functional
import json

from models.seq2seq_tds import TDSSeq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training.tensorboard_callback import TensorBoardCallback
import torch

LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-e", "--experiment_name", default="default", type=click.STRING, help="Experiment name.")
@click.option("-c", "--config_path", default="config/train.yaml", type=click.STRING,
              help="Path to train configurations.")
@click.option("-l", "--log_dir", default="log/", type=click.STRING, help="Log directory for tensorboard.")
def main(experiment_name, config_path, log_dir):
    Experiment.create(experiment_name)
    config = parse_yaml(config_path)["train"]
    config['decoder']['vocab_dim'] = len(config['labels'])
    train_loader, val_loader = create_data_loaders(**config)

    def cross_entropy_loss(batch, model):
        batch_x, batch_y, input_lengths, target_lengths = batch
        # check out the _collate_fn in loader to understand the next transformations
        batch_x = batch_x.squeeze(1).transpose(1, 2)
        batch_y = torch.split(batch_y, target_lengths.tolist())

        # max_len = max(batch_y, key=lambda x: x.size()[0]).size()[0]
        # padded_batch_y = []
        # for y in batch_y:
        #    padded_y = torch_functional.pad(y, (0, max_len - y.size()[0]))
        #    padded_batch_y.append(padded_y)
        # batch_y = torch.stack(padded_batch_y).type(torch.LongTensor)
        # batch_y = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first=True).type(torch.LongTensor)
        model_output, lens, loss = model(batch_x, input_lengths, batch_y)
        # loss = torch_functional.cross_entropy(model_output.permute(0, 2, 1), batch_y)
        return loss, (model_output, lens)

    model = TDSSeq2Seq(config['labels'], config["encoder"], config["decoder"])
    trainer = ModelTrainer(model, loss=cross_entropy_loss, epochs=config["max_epochs"],
                           train_data_loader=train_loader, val_data_loader=val_loader,
                           lr=config["learning_rate"], custom_model_eval=True,
                           metrics=[word_error_rate, character_error_rate],
                           decoder=GreedyDecoder(config['labels']),
                           callbacks=[TensorBoardCallback(log_dir)])

    try:
        trainer.start_training()
    except KeyboardInterrupt:
        trainer.stop_training()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
