import logging
import click
import torch

from sonosco.models.seq2seq_tds import TDSSeq2Seq
from sonosco.common.constants import SONOSCO
from sonosco.model.deserializer import ModelDeserializer
from sonosco.common.path_utils import parse_yaml
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training import Experiment, ModelTrainer
from sonosco.datasets import create_data_loaders
from sonosco.decoders import GreedyDecoder
from sonosco.training.evaluator import ModelEvaluator

LOGGER = logging.getLogger(SONOSCO)

def evaluate_deepspeech():
    path_to_model_checkpoint = '/Users/florianlay/roboy/sonosco/pretrained/model_checkpoint.pt'
    config_path = "../sonosco/config/train_seq2seq_tds_yuriy.yaml"

    config = parse_yaml(config_path)["train"]

    experiment = Experiment.create(config, LOGGER)

    model_deserializer = ModelDeserializer()

    model = model_deserializer.deserialize_model(cls=TDSSeq2Seq, path=path_to_model_checkpoint)

    train_loader, val_loader = create_data_loaders(**config)

    metrics = [word_error_rate, character_error_rate]

    evaluator = ModelEvaluator(model=model,
                               dataloader=val_loader,
                               bootstrap_size=100,
                               num_bootstraps=20,
                               decoder=GreedyDecoder(config["decoder"]['labels']),
                               metrics = metrics)

    evaluator.start_evaluation()
    evaluator.dump_evaluation(output_path=experiment.logs_path)

if __name__ == '__main__':
    evaluate_deepspeech()
