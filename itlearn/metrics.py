import math
import ipdb
import random
import numpy as np
import _pickle as pickle
import revtok
import os
from itertools import groupby
import getpass
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from torchtext import data, datasets
from collections import OrderedDict
import fractions

class Metrics:

    def __init__(self, name, *metrics, data_type="sum"): # data_type : sum, avg
        self.count = 0
        self.metrics = OrderedDict((metric, 0) for metric in metrics)
        self.name = name
        self.data_type = data_type

    def accumulate(self, count, *values):
        self.count += count
        for value, metric in zip(values, self.metrics):
            if self.data_type == "sum":
                self.metrics[metric] += value
            elif self.data_type == "avg":
                self.metrics[metric] += value * count

        return values[0] # loss

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key] / (self.count + 1e-9)
        raise AttributeError

    def __repr__(self):
        return ("{}: ".format(self.name) +
               "[{}]".format( ', '.join(["{}: {:.4f}".format(metric, getattr(self, metric))
                                         for metric, value in self.metrics.items() if value is not 0])))

    def reset(self):
        self.count = 0
        self.metrics.update({metric: 0 for metric in self.metrics})

class Best:
    def __init__(self, cmp_fn, *metrics, model=None, opt=None, path='', gpu=0, which=[0], debug=False, save="_best.pt"):
        self.cmp_fn = cmp_fn
        self.model = model
        self.opt = opt
        self.path = path + save
        self.metrics = OrderedDict((metric, None) for metric in metrics)
        self.gpu = gpu
        self.which = which
        self.best_cmp_value = None
        self.debug = debug

    def accumulate(self, *other_values):

        with torch.cuda.device(self.gpu):
            cmp_values = [other_values[which] for which in self.which]
            if self.best_cmp_value is None or \
               self.cmp_fn(self.best_cmp_value, *cmp_values) != self.best_cmp_value:
                self.metrics.update( { metric: value for metric, value in zip(
                    list(self.metrics.keys()), other_values) } )
                self.best_cmp_value = self.cmp_fn( [ list(self.metrics.items())[which][1] for which in self.which ] )

                if not self.debug and self.model is not None:
                    torch.save(self.model.state_dict(), self.path)

                if not self.debug and self.opt is not None:
                    torch.save([self.iters, self.opt.state_dict()], self.path + '.states')

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        raise AttributeError

    def __repr__(self):
        return ("BEST: " +
                ', '.join(["{}: {:.2f}".format(metric, value) for metric, value in self.metrics.items() if value is not 0]))

