import os
import wget
import yaml
import codecs


def try_create_directory(path: str) -> None:
    """
    tries to create a directory at given path
    Args:
        path:

    """
    if not os.path.exists(path):
        os.makedirs(path)


def try_download(destination: str, url: str) -> None:
    """
    Tries to download to destination from url
    Args:
        destination:
        url:

    """
    if not os.path.exists(destination):
        wget.download(url, destination)


def parse_yaml(file_path: str) -> dict:
    """
    load yaml into memory
    Args:
        file_path:

    Returns: dict with the yaml file content

    """
    with codecs.open(file_path, "r", "utf-8") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
