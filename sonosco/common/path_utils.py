import os
import wget


def try_create_directory(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def try_download(destination: str, url: str):
    if not os.path.exists(destination):
        wget.download(url, destination)
