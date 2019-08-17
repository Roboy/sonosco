import logging
import click
import torch

from sonosco.models.seq2seq_tds import TDSSeq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.decoders import GreedyDecoder, BeamCTCDecoder
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training.tensorboard_callback import TensorBoardCallback
from sonosco.training.model_checkpoint import ModelCheckpoint
from sonosco.config.global_settings import CUDA_ENABLED


LOGGER = logging.getLogger(SONOSCO)


@click.command()
@click.option("-e", "--experiment_name", default="default", type=click.STRING, help="Experiment name.")
@click.option("-c", "--config_path", default="../sonosco/config/train_seq2seq_tds.yaml", type=click.STRING,
              help="Path to train configurations.")
def main(experiment_name, config_path):
    experiment = Experiment.create(experiment_name)
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

    trainer = ModelTrainer(model, loss=cross_entropy_loss, epochs=config["max_epochs"],
                           train_data_loader=train_loader, val_data_loader=val_loader,
                           lr=config["learning_rate"], custom_model_eval=True,
                           metrics=[word_error_rate, character_error_rate],
                           decoder=GreedyDecoder(config['labels']),
                           device=device,
                           callbacks=[TensorBoardCallback(experiment.plots_path),
                                      ModelCheckpoint(experiment.checkpoints_path)])

    try:
        trainer.start_training()
    except KeyboardInterrupt:
        trainer.stop_training()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
