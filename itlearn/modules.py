import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.bleu import sentence_bleu
import random
from utils import xlen_to_inv_mask, cuda, take_last

class ArgsModule(torch.nn.Module):
    def __init__(self, args):
        super(ArgsModule, self).__init__()
        args_dict = vars(args) # Save argparse arguments into member variables
        for key in args_dict:
            setattr(self, key, args_dict[key])

    def forward(self):
        return
