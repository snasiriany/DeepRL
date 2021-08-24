#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import numpy as np
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *


def get_logger(env_name, tag='default', log_level=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    import pathlib
    import os.path as osp
    base_logdir = pathlib.Path(__file__).parent.parent.parent.resolve()
    exp_logdir = osp.join(base_logdir, 'data', env_name, '%s-%s' % (tag, get_time_str()))
    for path in [
        exp_logdir,
        osp.join(exp_logdir, 'model'),
        osp.join(exp_logdir, 'tf_log'),
    ]:
        if not os.path.exists(path):
            os.makedirs(path)

    if tag is not None:
        fh = logging.FileHandler(osp.join(exp_logdir, 'log.txt'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return Logger(logger, exp_logdir, log_level)


class Logger(object):
    def __init__(self, vanilla_logger, log_dir, log_level=0):
        self.log_level = log_level
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir + '/tf_log')
        if vanilla_logger is not None:
            self.info = vanilla_logger.info
            self.debug = vanilla_logger.debug
            self.warning = vanilla_logger.warning
        self.all_steps = {}

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def add_scalar(self, tag, value, step=None, log_level=0):
        if log_level > self.log_level:
            return
        value = self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step=None, log_level=0):
        if log_level > self.log_level:
            return
        values = self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step)
