import torch
from torch.nn import functional as F
from metrics import Metrics
from misc.bleu import computeBLEU, print_bleu
import sys
import torch
import os
from run_utils import get_model, get_data
from hyperparams import Params
from tqdm import tqdm


def valid_model(args, model, dev_it, dev_metrics, decode_method, beam_width=5, test_set="valid"):
    with torch.no_grad():
        model.eval()
        src_corpus, trg_corpus, hyp_corpus = [], [], []

        for j, dev_batch in tqdm(enumerate(dev_it)):
            if args.dataset == "iwslt" or args.dataset == 'iwslt_small':
                src, src_len = dev_batch.src
                trg, trg_len = dev_batch.trg
            elif args.dataset == "multi30k":
                src_lang, trg_lang = args.pair.split("_")
                src, src_len = dev_batch.__dict__[src_lang]
                trg, trg_len = dev_batch.__dict__[trg_lang]

            logits, _ = model(src[:,1:], src_len-1, trg[:,:-1])
            nll = F.cross_entropy(logits, trg[:,1:].contiguous().view(-1), size_average=True, ignore_index=0, reduce=True)
            num_trg = (trg[:,1:] != 0).sum().item()

            dev_metrics.accumulate(num_trg, nll.item())
            hyp = model.decode(src, src_len, decode_method, beam_width)
            src_corpus.extend( args.src.reverse( src ) )
            trg_corpus.extend( args.trg.reverse( trg ) )
            hyp_corpus.extend( args.trg.reverse( hyp ) )

        bleu = computeBLEU(hyp_corpus, trg_corpus, corpus=True)
        print(dev_metrics)
        print("{} {} : {}".format(test_set, decode_method, print_bleu(bleu)))
    return bleu


home_path = os.path.dirname(os.path.abspath(__file__))

# Parse from cmd to get JSON
parsed_args, _ = Params.parse_cmd(sys.argv)
if 'data_dir' not in parsed_args:
    raise ValueError('You must provide data_dir')
if 'ckpt' not in parsed_args:
    raise ValueError('You must provide ckpt')
if 'config' in parsed_args:
    print('Find json_config')
    args = Params(parsed_args['config'])
else:
    raise ValueError('You must pass in --config!')

# Update some of them with command line
args.update(parsed_args)
_, dev_it = get_data(args)
model = get_model(args)
if args.gpu > -1 and torch.cuda.device_count() > 0:
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = lambda storage, loc: storage
pretrained = torch.load(args.ckpt, map_location)
model.load_state_dict(pretrained)
print("Pretrained model loaded.")
dev_metrics = Metrics('dev_loss', 'nll', data_type="avg")
print(args.dataset)
dev_bleu = valid_model(args, model, dev_it, dev_metrics, 'greedy')
