import yaml
import torch

from typing import Dict, Any
from torch import device
import os

def get_config(path='config.yaml'):
    return yaml.load(open(path), Loader=yaml.FullLoader)


def transcribe(model_config: Dict[str, Any], audio_path: str, device: device) -> str:
    spect = model_config['processor'].parse_audio(audio_path)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model_config['model'](spect, input_sizes)
    transcription, offsets = model_config['decoder'].decode(out, output_sizes)
    return transcription[0][:4]

def create_pseudo_db(config=get_config()):
    db_path = f"{os.path.dirname(os.path.realpath(__file__))}/{config['data_base_path']}"
    if not os.path.exists(db_path):
        os.mkdir(db_path)
