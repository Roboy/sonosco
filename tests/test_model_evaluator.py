import logging
import torch

from sonosco.common.constants import SONOSCO
from sonosco.common.utils import setup_logging
from sonosco.common.path_utils import parse_yaml
from sonosco.training.word_error_rate import word_error_rate
from sonosco.training.character_error_rate import character_error_rate
from sonosco.training import Experiment
from sonosco.decoders import GreedyDecoder
from sonosco.training.evaluator import ModelEvaluator
from sonosco.models import DeepSpeech2
from common.global_settings import CUDA_ENABLED
from torch.utils.data import RandomSampler
from sonosco.datasets.processor import AudioDataProcessor
from sonosco.datasets.loader import AudioDataLoader
from sonosco.datasets.dataset import AudioDataset

LOGGER = logging.getLogger(SONOSCO)

def evaluate_deepspeech():
    setup_logging(LOGGER)
    path_to_model_checkpoint = '/Users/florianlay/roboy/sonosco/pretrained/deepspeech_final.pth'
    config_path = "bootstrap_deepspeech.yaml"

    config = parse_yaml(config_path)

    experiment = Experiment.create(config, LOGGER)

    #model_deserializer = ModelDeserializer()

    #model = model_deserializer.deserialize_model(cls=TDSSeq2Seq, path=path_to_model_checkpoint)
    model = DeepSpeech2.load_model(path_to_model_checkpoint)
    model.eval()

    processor = AudioDataProcessor(**config)
    test_dataset = AudioDataset(processor, manifest_filepath=config["test_manifest"])
    sampler = RandomSampler(data_source=test_dataset, replacement=True,
                                       num_samples=2)
    test_loader = AudioDataLoader(dataset=test_dataset, num_workers=config["num_data_workers"], sampler=sampler)

    device = torch.device("cuda" if CUDA_ENABLED else "cpu")

    metrics = [word_error_rate, character_error_rate]
    evaluator = ModelEvaluator(model,
                               test_loader,
                               config['bootstrap_size'],
                               config['num_bootstraps'],
                               decoder=GreedyDecoder(config['labels']),
                               device=device,
                               metrics = metrics)

    evaluator.start_evaluation(log_path=experiment.logs)

if __name__ == '__main__':
    evaluate_deepspeech()
