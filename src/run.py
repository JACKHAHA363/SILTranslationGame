import sys
import torch
import numpy as np

from time import strftime, localtime
from os.path import join
import os
from pathlib import Path

from data import build_dataset
from utils import set_seed, get_logger
from agents import Agents
from agent import RNNAttn, ImageCaptioning, RNNLM, ImageGrounding
from hyperparams import Params, get_hp_str

home_path = os.path.dirname(os.path.abspath(__file__))
main_path = os.path.dirname(home_path)

# Parse from cmd to get JSON
parsed_args, _ = Params.parse_cmd(sys.argv)

if 'config' in parsed_args:
    print('Find json_config')
    args = Params(parsed_args['config'])
else:
    raise ValueError('You must pass in --config!')

if 'exp_dir' in parsed_args:
    print('exp_dir', parsed_args['exp_dir'])
else:
    raise ValueError('You must pass in the --exp_dir')

# Update some of them with command line
args.update(parsed_args)
args.exp_dir = os.path.abspath(args.exp_dir)

folders = ["event", "model", "log", "param"]
if args.setup in ['single', 'joint']:
    folders.append('decoding')

if args.mode == "train":
    for name in folders:
        folder = "{}/{}_{}/".format(name, args.date, args.experiment) \
                if hasattr(args, "date") and hasattr(args, "experiment") else name + '/'
        args.__dict__["{}_path".format(name)] = os.path.join(args.exp_dir, folder)
        if not args.debug:
            Path(args.__dict__["{}_path".format(name)]).mkdir(parents=True, exist_ok=True)

    args.prefix = strftime("%m.%d_%H.%M.", localtime())
    args.hp_str = get_hp_str(args)
    args.id_str = args.prefix + "_" + args.hp_str

logger = get_logger(args)

# for decoding : load the same model architecture from the json file
if args.mode == "test" and hasattr(args, "experiment") and hasattr(args, "id_str"):
    experiment = args.experiment
    id_str = args.id_str
    cpt_iter = args.cpt_iter if hasattr(args, 'cpt_iter') else None
    dataset = args.dataset

    load_param_from = "{}/param/{}/{}".format(main_path, experiment, id_str)
    args.update(load_param_from)
    args.load_from = "{}/model/{}/{}".format(main_path, experiment, id_str)
    args.mode = "test"
    args.debug = True
    args.dataset = dataset
    logger.info("Hyperparameters loaded.")
    args.batch_size = 400

set_seed(args)

if args.mode == "train" and not args.debug:
    args.save((str(args.param_path + args.id_str)))

if args.setup == "ranker":
    train_it, dev_it = {}, {}
    for dataset in ["coco", "multi30k"]:
        train_it_, dev_it_ = build_dataset(args, dataset)
        train_it[dataset] = train_it_
        dev_it[dataset] = dev_it_
else:
    train_it, dev_it = build_dataset(args, args.dataset)

args.__dict__.update({'logger': logger})

if args.mode == "train":
    args.logger.info(args)
    args.logger.info('Starting with HPARAMS: {}'.format(args.hp_str))

if args.setup == "joint":
    model = Agents(args)
elif args.setup == "single":
    model = RNNAttn(args, args.voc_sz_src, args.voc_sz_trg)
elif args.setup == "lm":
    model = RNNLM(args, len(args.EN.vocab.itos))
elif args.setup == "ranker":
    if args.img_pred_loss == "nll":
        model = ImageCaptioning(args, len(args.EN.vocab.itos))
    else:
        model = ImageGrounding(args, len(args.EN.vocab.itos))

if args.gpu > -1 and torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = lambda storage, loc: storage

model_path = join(main_path, 'model/{}/{}_best.pt')
# Loading EN LM
extra_input = {"en_lm":None, "img":{"multi30k":[None,None]}, "ranker":None}
if args.setup == "joint" and args.use_en_lm:
    from pretrained_models import en_lm
    experiment = en_lm[args.en_lm_dataset]['experiment']
    pt_model = en_lm[args.en_lm_dataset][args.aux_size]
    load_param_from = "{}/param/{}/{}".format(main_path, experiment, pt_model)
    args.logger.info( "Loading LM from: " + load_param_from )
    args_ = Params()
    args_.update(load_param_from)
    LM_CLS = ImageCaptioning if args.en_lm_dataset in ["coco", "multi30k"] else RNNLM
    en_lm = LM_CLS(args_, len(args.EN.vocab.itos))
    en_lm.load_state_dict( torch.load(model_path.format(experiment, pt_model), map_location) )
    en_lm.eval()
    if torch.cuda.is_available():
        en_lm.cuda(args.gpu)
    extra_input["en_lm"] = en_lm

#if False and ( args.setup == "ranker" or (args.setup == "joint" and args.use_ranker) ):
if args.setup == "ranker" or (args.setup == "joint" and args.use_ranker) :
    if args.setup == "joint" and args.use_ranker:
        from pretrained_models import ranker
        experiment = ranker[args.ranker_dataset][args.img_pred_loss]['experiment']
        pt_model = ranker[args.ranker_dataset][args.img_pred_loss][args.D_img]
        load_param_from = "{}/param/{}/{}".format(main_path, experiment, pt_model)
        args.logger.info( "Loading ranker from: " + load_param_from )
        args_ = Params()
        args_.update(load_param_from)
        if args.img_pred_loss == "nll":
            ranker = ImageCaptioning(args_, len(args.EN.vocab.itos))
        else:
            ranker = ImageGrounding(args_, len(args.EN.vocab.itos))
        ranker.load_state_dict( torch.load(model_path.format(experiment, pt_model), map_location) )
        ranker.eval()
        if torch.cuda.is_available():
            ranker.cuda(args.gpu)
        extra_input["ranker"] = ranker

    img = {}
    if args.setup == "ranker" or "multi30k" in args.dataset:
        size = "resnet152/" if args.D_img == 2048 else "resnet34/"
        img_path = "/private/home/jasonleeinf/corpora/multi30k/images/{}".format(size)
        img["multi30k"] = [torch.load(img_path+x).cpu() for x in ["train_feats.pt", "valid_feats.pt"]]
        args.logger.info("Loading {} image features: train {} valid {}".format( \
                          args.dataset.upper(), img['multi30k'][0].shape, img['multi30k'][1].shape ))
    if args.setup == "ranker" or "coco" in args.dataset:
        size = "resnet152/" if args.D_img == 2048 else "resnet34/"
        img_path = "/private/home/jasonleeinf/corpora/coco/feats/{}".format(size)
        img["coco"] = [torch.load(img_path+x) for x in ["train_feats.pt", "valid_feats.pt"]]
        args.logger.info("Loading {} image features: train {} valid {}".format( \
                          args.dataset.upper(), img['coco'][0].shape, img['coco'][1].shape ))
    extra_input["img"] = img

# Loading checkpoints pretrained on IWSLT
if args.setup == "joint":
    if args.model == "RNNAttn":
        en_de = ("180816_04_en_de", "08.17_01.38._single_rnnattn_en_de_emb256_hid256_1l_lr3e-04_linear_ann500k_drop0.3_clip0.1_")
        fr_en = ("180817_01_fr_en", "08.17_21.52._single_rnnattn_fr_en_emb256_hid256_1l_lr3e-04_linear_ann500k_drop0.2_clip0.1_")
    elif args.model == "RNN":
        en_de = ("180906_02_en_de_noattn_big", "09.06_01.30._single_rnn_en_de_emb512_hid512_1l_lr1e-04_linear_ann1500k_drop0.5_clip1.0_")
        fr_en = ("180906_03_fr_en_noattn_big", "09.06_01.33._single_rnn_fr_en_emb512_hid512_1l_lr3e-04_linear_ann1000k_drop0.6_clip1.0_")
    load_from_en_de = "{}/model/{}/{}".format(main_path, *en_de)
    load_from_fr_en = "{}/model/{}/{}".format(main_path, *fr_en)
    load_from_en_de = "{}_best.pt".format(load_from_en_de) if args.cpt_iter == "best" else "{}_iter={}.pt".format(load_from_en_de, args.cpt_iter)
    load_from_fr_en = "{}_best.pt".format(load_from_fr_en) if args.cpt_iter == "best" else "{}_iter={}.pt".format(load_from_fr_en, args.cpt_iter)

    if args.cpt_iter != 0:
        model.fr_en.load_state_dict( torch.load( load_from_fr_en, map_location ) )
        args.logger.info("Loading Fr -> En checkpoint : {}".format(load_from_fr_en))
        model.en_de.load_state_dict( torch.load( load_from_en_de, map_location ) )
        args.logger.info("Loading En -> De checkpoint : {}".format(load_from_en_de))

        if args.fix_fr2en:
            for param in list(model.fr_en.parameters()):
                param.requires_grad = False
            args.logger.info("Fixed FR->EN agent")
            #model.fr_en.dec.msg_len_ratio = -1.0 # make Fr->En agent predict its own length

if args.mode == "train":
    args.logger.info(str(model))

    params, param_names = [], []
    for name, param in model.named_parameters():
        params.append(param)
        param_names.append(name)
        args.logger.info("{} {} {}".format(name, param.size(), "-----FIXED-----" if not param.requires_grad else ""))

    args.logger.info("Model size {:,}".format( sum( [ np.prod(x.size()) for x in params ] )) )

if torch.cuda.is_available() and args.gpu > -1:
    model.cuda(args.gpu)

if args.mode == 'train':
    if args.setup == "single":
        from train_single import train_model
        train_model(args, model, (train_it, dev_it))

    elif args.setup == "joint":
        from train_joint import train_model
        train_model(args, model, (train_it, dev_it), extra_input)

    elif args.setup == "ranker":
        if args.img_pred_loss == "nll":
            #from train_captioner import train_model
            from train_captioner_flickr30k import train_model
            train_model(args, model)
        #elif args.img_pred_loss == "vse":
        #    from train_ranker import train_model
        elif args.img_pred_loss in ["vse", "mse"]:
            from train_raw_ranker import train_model
            #from train_raw_ranker_pretrained_mse import train_model
            train_model(args, model)

    elif args.setup == "lm":
        from train_lm import train_model
        train_model(args, model, (train_it, dev_it))

elif args.mode == 'test':
    if args.setup == "single":
        from decode_single import decode_model
    elif args.setup == "joint":
        from decode_joint import decode_model

    decode_model(args, model, (dev_it))

args.logger.info("done.")
