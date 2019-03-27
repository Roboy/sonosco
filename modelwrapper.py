import os.path
from datetime import datetime

import torch

from models.deepspeech2 import DeepSpeech2

models = {"deepspeech2": DeepSpeech2}


class ModelWrapper(object):
    DEF_PATH = "examples/checkpoints/"

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", models["deepspeech2"])

        if kwargs.get("continue"):
            path = kwargs.get("from", ModelWrapper.get_default_path())
            self.model.package = torch.load(path, map_location=lambda storage, loc: storage)
            self.model.load_model(path)

        self.save_path = kwargs.get("save", ModelWrapper.DEF_PATH + str(datetime.now().timestamp()))

        self.cuda = kwargs.get("cuda")
        self.apex = kwargs.get("apex") if self.cuda else False
        self.half = self.apex if self.apex else kwargs.get("half")

    def train(self):
        pass

    def test(self):
        pass

    def infer(self, sound):
        pass

    @staticmethod
    def get_default_path(def_path: str = DEF_PATH) -> str:
        """
        Returns the path to the latest checkpoint in the default location
        :param def_path: default path where checkpoints are stored
        :return: the path to the latest checkpoint
        """
        latest_subdir = max([os.path.join(def_path, d) for d in os.listdir(def_path)], key=os.path.getmtime)
        default = latest_subdir + "/final.pth"
        return default


    def print_training_info(self, epoch, loss, cer, wer):
        print("\nTraining Information")
        print(f"- Epoch:\t{epoch}")
        print(f"- Current Loss:\t{loss}")
        print(f"- Current CER: \t{cer}")
        print(f"- Current WER: \t{wer}")
