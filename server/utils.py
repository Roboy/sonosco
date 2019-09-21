import pathlib
import yaml
import os

from datetime import datetime
from sonosco.common.path_utils import try_create_directory


def get_config(path='config.yaml'):
    return yaml.load(open(path), Loader=yaml.FullLoader)


def create_pseudo_db(db_path='~/.sonosco/audio_data/'):
    db_path = os.path.expanduser(db_path)
    pathlib.Path(db_path).mkdir(parents=True, exist_ok=True)
    return db_path


def create_session_dir(sonosco_home):
    sesson_dir = os.path.join(sonosco_home, "sessions", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    try_create_directory(sesson_dir)
    return sesson_dir


