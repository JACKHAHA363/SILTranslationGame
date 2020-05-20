""" Running agents, language model, and image grounding pretraining
"""
import sys
import torch
import numpy as np
from time import strftime, localtime
import os
from pathlib import Path

from run_utils import get_model, get_data, get_ckpt_paths
from utils.misc import set_seed, get_logger
from utils.hyperparams import Params, get_hp_str
home_path = os.path.dirname(os.path.abspath(__file__))

# Parse from cmd to get JSON
parsed_args, _ = Params.parse_cmd(sys.argv)

if 'config' in parsed_args:
    print('Find json_config')
    args = Params(parsed_args['config'])
else:
    raise ValueError('You must pass in --config!')

# Update some of them with command line
args.update(parsed_args)

if not hasattr(args, 'exp_dir'):
    raise ValueError('You must provide exp_dir')
if not hasattr(args, 'data_dir'):
    raise ValueError('You must provide data_dir')

args.exp_dir = os.path.abspath(args.exp_dir)
main_path = args.exp_dir

folders = ["event", "model", "log", "param"]
if args.setup in ['single']:
    folders.append('decoding')

for name in folders:
    folder = "{}/{}/".format(name, args.experiment) if hasattr(args, "experiment") else name + '/'
    args.__dict__["{}_path".format(name)] = os.path.join(args.exp_dir, folder)
    Path(args.__dict__["{}_path".format(name)]).mkdir(parents=True, exist_ok=True)

if not hasattr(args, 'hp_str'):
    args.hp_str = get_hp_str(args)
    args.prefix = strftime("%m.%d_%H.%M.", localtime())
    args.id_str = args.prefix + "_" + args.hp_str
logger = get_logger(args)
set_seed(args)

# Save config
args.save((str(args.param_path + args.id_str)))

# Data
train_it, dev_it = get_data(args)

args.__dict__.update({'logger': logger})
args.logger.info(args)
args.logger.info('Starting with HPARAMS: {}'.format(args.hp_str))

# Model
model = get_model(args)
extra_input = {}
if args.gpu > -1 and torch.cuda.device_count() > 0:
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = lambda storage, loc: storage

resume_path = os.path.join(args.model_path, args.id_str + '_latest.pt')
if os.path.exists(resume_path):
    extra_input['resume'] = torch.load(resume_path, map_location)
    args.logger.info('Resume training from iter={}'.format(extra_input['resume']['iters']))
    model.load_state_dict(extra_input['resume']['model'])

args.logger.info(str(model))

# Params
params, param_names = [], []
for name, param in model.named_parameters():
    params.append(param)
    param_names.append(name)
    args.logger.info("{} {} {}".format(name, param.size(), "-----FIXED-----" if not param.requires_grad else ""))

args.logger.info("Model size {:,}".format( sum( [ np.prod(x.size()) for x in params ] )) )

if torch.cuda.device_count() > 0 and args.gpu > -1:
    model.cuda(args.gpu)

# Main
if args.setup == "single":
    from pretrain.train_single import train_model
    train_model(args, model, (train_it, dev_it))

elif args.setup == "ranker":
    if args.img_pred_loss == "nll":
        from pretrain.train_captioner import train_model
        train_model(args, model, (train_it, dev_it), extra_input)
    elif args.img_pred_loss in ["vse", "mse"]:
        from pretrain.train_raw_ranker import train_model
        train_model(args, model)

elif args.setup == "lm":
    from pretrain.train_lm import train_model
    train_model(args, model, (train_it, dev_it))

args.logger.info("done.")
