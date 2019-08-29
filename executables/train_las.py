import logging
import click
import torch

from sonosco.models.seq2seq_las import Seq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.decoders import GreedyDecoder
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training.losses import cross_entropy_loss
from sonosco.training.tb_text_comparison_callback import TbTextComparisonCallback
from sonosco.training.disable_soft_window_attention import DisableSoftWindowAttention
from sonosco.training.tb_teacher_forcing_text_comparison_callback import TbTeacherForcingTextComparisonCallback
from sonosco.config.global_settings import CUDA_ENABLED

LOGGER = logging.getLogger(SONOSCO)

EOS = '$'
SOS = '#'
PADDING_VALUE = '%'


@click.command()
@click.option("-c", "--config_path", default="../sonosco/config/train_seq2seq_las.yaml",
              type=click.STRING, help="Path to train configurations.")
def main(config_path):
    config = parse_yaml(config_path)["train"]
    experiment = Experiment.create(config, LOGGER)

    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    char_list = config["labels"] + EOS + SOS

    config["decoder"]["vocab_size"] = len(char_list)
    config["decoder"]["sos_id"] = char_list.index(SOS)
    config["decoder"]["eos_id"] = char_list.index(EOS)

    # Create model
    model = Seq2Seq(config["encoder"], config["decoder"])
    model.to(device)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(**config)

    # Create model trainer
    trainer = ModelTrainer(model, loss=cross_entropy_loss, epochs=config["max_epochs"],
                           train_data_loader=train_loader, val_data_loader=val_loader,
                           lr=config["learning_rate"], weight_decay=config['weight_decay'],
                           metrics=[word_error_rate, character_error_rate],
                           decoder=GreedyDecoder(config['labels']),
                           device=device, test_step=config["test_step"], custom_model_eval=True)

    trainer.add_callback(TbTextComparisonCallback(log_dir=experiment.plots_path))
    trainer.add_callback(TbTeacherForcingTextComparisonCallback(log_dir=experiment.plots_path))
    trainer.add_callback(DisableSoftWindowAttention())

    # Setup experiment with a model trainer
    experiment.setup_model_trainer(trainer, checkpoints=True, tensorboard=True)

    try:
        experiment.start()
    except KeyboardInterrupt:
        experiment.stop()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
