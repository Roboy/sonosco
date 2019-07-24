import json
import yaml

def get_config(path='config.yaml'):
    return yaml.load(open(path), Loader=yaml.FullLoader)

