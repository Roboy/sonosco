import logging
import os
import numpy as np


def setup_logging(logger: logging.Logger, filename=None, verbosity=False):
    logger.setLevel(logging.DEBUG)
    if filename is not None:
        log_directory = os.path.dirname(filename)
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        filename = os.path.join(log_directory, f"{filename}.log")
        f_handler = logging.FileHandler(filename=filename, mode="w")
        f_handler.setLevel(logging.DEBUG)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG) if verbosity else c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)


def random_float(low: float, high: float):
    return np.random.random() * (high - low) + low
