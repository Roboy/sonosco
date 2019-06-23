import os
import wget
import yaml
import codecs


def try_create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def try_download(destination: str, url: str):
    if not os.path.exists(destination):
        wget.download(url, destination)


def parse_yaml(file_path: str):
    with codecs.open(file_path, "r", "utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
