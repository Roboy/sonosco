import argparse
import os
import wave
from typing import Dict

import yaml

from modelwrapper import ModelWrapper

parser = argparse.ArgumentParser(description='ASR inference')
parser.add_argument('--config', metavar='DIR',
                    help='Path to inference config file', default='config/infer.yaml')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file)
    config_dict: Dict = config["infer"]
    model = ModelWrapper(**config_dict)
    if "wave_path" in config_dict.keys() and os.path.isfile(config_dict.get("wave_path")):
        sound = wave.open(config_dict.get("wave_path"))
        print(model.infer(sound))
    else:
        print("Wave file not found!")
