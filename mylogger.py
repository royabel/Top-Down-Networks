import logging
from pathlib import Path
import os

import torch


def setup_logger(logger, fname):
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(format=('%(asctime)s ' + '%(message)s'))
    logger.level = logging.INFO
    Path('logs').mkdir(exist_ok=True)

    # logger_handler = logging.FileHandler(os.path.join('logs', fname))
    # logger_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    # logger.addHandler(logger_handler)
    # logger.addHandler(logging.StreamHandler())

    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(os.path.join('logs', fname))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    logger.info('PyTorch version: %s' % torch.__version__)
    logger.info('cuDNN enabled: %s' % torch.backends.cudnn.enabled)
