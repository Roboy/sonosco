import logging
import numpy as np
import os
import subprocess
import os.path as path

from shutil import copyfile
from typing import Tuple


def setup_logging(logger: logging.Logger, filename=None, verbosity=False):
    logger.setLevel(logging.DEBUG) if verbosity else logger.setLevel(logging.INFO)

    if filename is not None:
        add_log_file(filename, logger)

    # Somehow stream handler is already setup before this or after at some different place
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG) if verbosity else c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)


def add_log_file(filename: str, logger: logging.Logger):
    log_directory = os.path.dirname(filename)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    filename = os.path.join(log_directory, f"{filename}.log")
    f_handler = logging.FileHandler(filename=filename, mode="w")
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)


def random_float(low: float, high: float):
    return np.random.random() * (high - low) + low


def copy_code(source_dir, dest_dir, exclude_dirs: Tuple[str] = tuple(), exclude_files: Tuple[str] = tuple()):
    """
    Copies code from source_dir to dest_dir. Excludes specified folders and files by substring-matching.
    Parameters:
        source_dir (string): location of the code to copy
        dest_dir (string): location where the code should be copied to
        exclude_dirs (list of strings): folders containing strings specified in this list will be ignored
        exclude_files (list of strings): files containing strings specified in this list will be ignored
    """
    source_basename = path.basename(source_dir)
    for root, dirs, files in os.walk(source_dir, topdown=True):

        # skip ignored dirs
        if any(ex_subdir in root for ex_subdir in exclude_dirs):
            continue

        # construct destination dir
        cropped_root = root[2:] if (root[:2] == './') else root
        subdir_basename = path.basename(cropped_root)

        # do not treat the root as a subdir
        if subdir_basename == source_basename:
            subdir_basename = ""
        dest_subdir = os.path.join(dest_dir, subdir_basename)

        # create destination folder
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        # copy files
        for filename in filter(lambda x: not any(substr in x for substr in exclude_files), files):
            source_file_path = os.path.join(root, filename)
            dest_file_path = os.path.join(dest_subdir, filename)
            copyfile(source_file_path, dest_file_path)


def retrieve_git_hash():
    """
    Retrieves and returns the current gith hash if execution location is a git repo.
    """
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return git_hash
    except subprocess.CalledProcessError as e:
        print(e.output)
    return False


def save_run_params_in_file(folder_path, run_config):
    """
    Receives a run_config class, retrieves all member variables and saves them
    in a config file for logging purposes.
    Parameters:
        folder_path - output folder
        filename - output filename
        run_config - shallow class with parameter members
    """
    with open(path.join(folder_path, "run_params.conf"), 'w') as run_param_file:
        for attr, value in sorted(run_config.__dict__.items()):
            run_param_file.write(f"{attr}: {value}\n")


def labels_to_dict(labels: str):
    return dict([(labels[i], i) for i in range(len(labels))])


def reverse_labels_to_dict(labels: str):
    return dict([(i, c) for (i, c) in enumerate(labels)])
