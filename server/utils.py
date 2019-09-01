import pathlib

import yaml
import torch

from typing import Dict, Any
from torch import device
import os
from datetime import datetime
from sonosco.common.path_utils import try_create_directory


def get_config(path='config.yaml'):
    return yaml.load(open(path), Loader=yaml.FullLoader)


def transcribe(model_config: Dict[str, Any], audio_path: str, device: device) -> str:
    spect = model_config['processor'].parse_audio_from_file(audio_path)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model_config['model'](spect, input_sizes)
    transcription, offsets = model_config['decoder'].decode(out, output_sizes)
    return transcription[0][:4]


def create_pseudo_db(db_path='~/.sonosco/audio_data/'):
    db_path = os.path.expanduser(db_path)
    pathlib.Path(db_path).mkdir(parents=True, exist_ok=True)
    return db_path


def create_session_dir(sonosco_home):
    sesson_dir = os.path.join(sonosco_home, "sessions", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    try_create_directory(sesson_dir)
    return sesson_dir


