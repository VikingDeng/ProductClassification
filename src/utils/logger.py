import logging
import os
import sys
from torch.utils.tensorboard import SummaryWriter

def setup_logger(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(work_dir, 'train.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    tb_writer = SummaryWriter(log_dir=os.path.join(work_dir, 'tf_logs'))
    return logger, tb_writer
