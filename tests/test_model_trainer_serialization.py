import logging
import torch
from sonosco.model.serializer import ModelSerializer

from sonosco.models.seq2seq_las import Seq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.common.path_utils import parse_yaml
from sonosco.training import ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.decoders import GreedyDecoder
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training.losses import cross_entropy_loss
from sonosco.config.global_settings import CUDA_ENABLED
from sonosco.model.deserializer import ModelDeserializer

LOGGER = logging.getLogger(SONOSCO)

EOS = '$'
SOS = '#'
PADDING_VALUE = '%'

def test_mode_trainer_serialization():
    config_path = "model_trainer_config_test.yaml"
    config = parse_yaml(config_path)["train"]

    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    char_list = config["labels"] + EOS + SOS

    config["decoder"]["vocab_size"] = len(char_list)
    config["decoder"]["sos_id"] = char_list.index(SOS)
    config["decoder"]["eos_id"] = char_list.index(EOS)

    # Create mode
    if config.get('checkpoint_path'):
        LOGGER.info(f"Loading model from checkpoint: {config['checkpoint_path']}")
        loader = ModelDeserializer()
        model = loader.deserialize(Seq2Seq, config["checkpoint_path"])
    else:
        model = Seq2Seq(config["encoder"], config["decoder"])
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
    loader = ModelDeserializer()
    s = ModelSerializer()
    s.serialize(trainer, '/Users/w.jurasz/ser')
    trainer_deserialized = loader.deserialize(ModelTrainer, '/Users/w.jurasz/ser', {
        'train_data_loader': train_loader,
        'val_data_loader': val_loader,
        'test_data_loader': test_loader,
    })
    assert trainer_deserialized is not None