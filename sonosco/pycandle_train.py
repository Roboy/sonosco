import torch
import torchvision
import torch.nn.functional as F

from sonosco.models.deepspeech2 import DeepSpeech2
from sonosco.pycandle.general.experiment import Experiment
from sonosco.pycandle.training.model_trainer import ModelTrainer


def load_datasets(batch_size_train, batch_size_test):
    pass

model = DeepSpeech2().cuda()
experiment = Experiment('mnist_example')
train_loader, val_loader = load_datasets(batch_size_train=64, batch_size_test=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model_trainer = ModelTrainer(model, optimizer, F.nll_loss, 20, train_loader, gpu=0)
model_trainer.start_training()
