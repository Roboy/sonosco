import argparse
from typing import Dict

import yaml

from modelwrapper import ModelWrapper

parser = argparse.ArgumentParser(description='ASR training')
parser.add_argument('--config', metavar='DIR',
                    help='Path to train config file', default='config/train.yaml')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file)
    config_dict: Dict = config["train"]
    model = ModelWrapper(**config_dict)
    model.train()
