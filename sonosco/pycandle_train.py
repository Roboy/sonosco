import torch
import torch.nn.functional as F
#import datasets.download_datasets.librispeech as librispeech

from datasets.AudioDataLoader import AudioDataLoader
from datasets.AudioDataSampler import BucketingSampler
from datasets.AudioDataset import AudioDataset
from models.deepspeech2 import DeepSpeech2
from pycandle.general.experiment import Experiment
from pycandle.training.model_trainer import ModelTrainer

def load_datasets(manifest_path, batch_size_train, batch_size_test):
    audio_conf = dict(sample_rate=16000,
                      window_size=.02,
                      window_stride=.01,
                      window='hamming')
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    test_dataset = AudioDataset(audio_conf=audio_conf, manifest_filepath=manifest_path, labels=labels,
                                normalize=False, augment=False)
    print("Dataset is created\n====================\n")

    batch_size = 16
    sampler = BucketingSampler(test_dataset, batch_size=batch_size)
    return AudioDataLoader(test_dataset, num_workers=4, batch_sampler=sampler)

# librispeech.main()
model = DeepSpeech2().cpu()
experiment = Experiment('mnist_example')
train_loader = load_datasets("./datasets/download_datasets/libri_test_clean_manifest.csv",
                                         batch_size_train=64, batch_size_test=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_trainer = ModelTrainer(model, optimizer, F.nll_loss, 20, train_loader, gpu=0)
model_trainer.start_training()
