import sys
import torch
import numpy as np
from time import strftime, localtime
import os
from pathlib import Path

from run_utils import get_model, get_data, get_ckpt_paths
from utils.misc import set_seed, get_logger
from models.agent import ImageCaptioning, RNNLM, ImageGrounding
from utils.hyperparams import Params, get_hp_str
from data import get_s2p_dataset
from finetune import SILTrainer, Trainer

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

assert args.setup in ['gumbel', 'gumbel_sil', 'a2c', 'a2c_sil']
folders = ["event", "model", "log", "param", "decoding", "misc"]

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


# Loading EN LM
extra_input.update({"en_lm": None, "img": {"multi30k": [None, None]}, "ranker": None, "s2p_it": None})
if args.use_en_lm:
    lm_param, lm_model = get_ckpt_paths(args.exp_dir, args.lm_ckpt)
    args.logger.info("Loading LM from: " + lm_param)
    args_ = Params(lm_param)
    LM_CLS = ImageCaptioning if args.en_lm_dataset in ["coco"] else RNNLM
    en_lm = LM_CLS(args_, len(args.EN.vocab.itos))
    en_lm.load_state_dict(torch.load(lm_model, map_location))
    en_lm.eval()
    if torch.cuda.device_count() > 0:
        en_lm.cuda(args.gpu)
    extra_input["en_lm"] = en_lm

if args.use_ranker:
    ranker_param, ranker_model = get_ckpt_paths(args.exp_dir, args.ranker_ckpt)
    args.logger.info("Loading ranker from: " + ranker_param)
    args_ = Params(ranker_param)
    if args.img_pred_loss == "nll":
        ranker = ImageCaptioning(args_, len(args.EN.vocab.itos))
    else:
        ranker = ImageGrounding(args_, len(args.EN.vocab.itos))
    ranker.load_state_dict(torch.load(ranker_model, map_location) )
    ranker.eval()
    if torch.cuda.device_count() > 0:
        ranker.cuda(args.gpu)
    extra_input["ranker"] = ranker

    img = {}
    flickr30k_dir = os.path.join(args.data_dir, 'flickr30k')
    train_feats = torch.load(os.path.join(flickr30k_dir, 'train_feat.pth'))
    val_feats = torch.load(os.path.join(flickr30k_dir, 'val_feat.pth'))
    img["multi30k"] = [torch.tensor(train_feats), torch.tensor(val_feats)]
    args.logger.info("Loading Flickr30k image features: train {} valid {}".format(
        img['multi30k'][0].shape, img['multi30k'][1].shape))
    extra_input["img"] = img

# Get iwslt its for s2p
extra_input['s2p_its'] = get_s2p_dataset(args)


# Loading checkpoints pretrained on IWSLT
if hasattr(args, 'en_de_ckpt') and args.en_de_ckpt is not None:
    _, en_de_model = get_ckpt_paths(args.exp_dir, args.en_de_ckpt, args.cpt_iter)
    model.en_de.load_state_dict(torch.load(en_de_model, map_location))
    args.logger.info("Loading En -> De checkpoint : {}".format(en_de_model))

if hasattr(args, 'fr_en_ckpt') and args.fr_en_ckpt is not None:
    _, fr_en_model = get_ckpt_paths(args.exp_dir, args.fr_en_ckpt, args.cpt_iter)
    model.fr_en.load_state_dict(torch.load(fr_en_model, map_location))
    args.logger.info("Loading Fr -> En checkpoint : {}".format(fr_en_model))
    if args.fix_fr2en:
        for param in list(model.fr_en.parameters()):
            param.requires_grad = False
        args.logger.info("Fixed FR->EN agent")

if torch.cuda.device_count() > 0 and args.gpu > -1:
    model.cuda(args.gpu)

# Main
if args.setup == "a2c" or args.setup == 'gumbel':
    trainer = Trainer(args, model, train_it, dev_it, extra_input)

elif args.setup == 'a2c_sil' or args.setup == 'gumbel_sil':
    trainer = SILTrainer(args, model, train_it, dev_it, extra_input)
else:
    raise ValueError
trainer.start()
args.logger.info("done.")
