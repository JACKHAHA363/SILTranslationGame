"""
Decode English from trained models
"""
import sys
import torch
from tqdm import tqdm
from time import strftime, localtime
import os
from pathlib import Path

from torchtext import data
from run_utils import get_model
from data import NormalField, NormalTranslationDataset, BOS, EOS, PAD, UNK, batch_size_fn, TextVocab, \
    TripleTranslationDataset
from utils import set_seed, get_logger
from hyperparams import Params, get_hp_str
from misc.bleu import computeBLEU, print_bleu

home_path = os.path.dirname(os.path.abspath(__file__))

# Parse from cmd to get JSON
parsed_args, _ = Params.parse_cmd(sys.argv)

if 'config' in parsed_args:
    print('Find json_config')
    args = Params(parsed_args['config'])
else:
    raise ValueError('You must pass in --config!')

# Update some of them with command line
if 'data_dir' not in parsed_args:
    raise ValueError('You must provide --data_dir')
if 'ckpt' not in parsed_args:
    raise ValueError('You must provide --ckpt')
if 'out' not in parsed_args:
    raise ValueError('You must provide --out')

args.update(parsed_args)

JOINT_SETUPS = ['joint', 'itlearn', 'gumbel', 'gumbel_itlearn', 'aug_s2p']
folders = ["event", "model", "log", "param"]
if args.setup in ['single'] + JOINT_SETUPS:
    folders.append('decoding')

# Data
def get_IWSLT(gpu, data_dir):
    device = "cuda:{}".format(gpu) if gpu > -1 and torch.cuda.device_count() > 0 else "cpu"
    bpe_path = os.path.join(data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"

    # IWSLT
    src_field = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                            include_lengths=True, batch_first=True)
    trg_field = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                            include_lengths=True, batch_first=True)
    vocabs = [fr_vocab, en_vocab]
    for (field, vocab) in zip([src_field, trg_field], vocabs):
        field.vocab = TextVocab(counter=torch.load(os.path.join(bpe_path, vocab)))

    src = '.fr'
    trg = '.en'
    pair = 'fr-en'
    dev_path = os.path.join(bpe_path, 'iwslt', pair, 'IWSLT16.TED.tst2013.' + pair)
    dev_data = NormalTranslationDataset(path=dev_path,
                                        exts=(src, trg), fields=(src_field, trg_field),
                                        load_dataset=False, save_dataset=False)
    iwslt_it = data.BucketIterator(dev_data, 256, device=device,
                                   batch_size_fn=batch_size_fn, train=False,
                                   repeat=False, shuffle=False,
                                   sort=False, sort_within_batch=True)
    return iwslt_it

def get_multi30k(args, gpu, data_dir):
    device = "cuda:{}".format(gpu) if gpu > -1 and torch.cuda.device_count() > 0 else "cpu"
    FR = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    EN = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    DE = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    bpe_path = os.path.join(data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    vocabs = [fr_vocab, en_vocab, de_vocab]
    fields = [FR, EN, DE]
    exts = ['.fr', '.en', '.de']

    for (field, vocab) in zip(fields, vocabs):
        field.vocab = TextVocab(counter=torch.load(os.path.join(bpe_path, vocab)))
    dev_data = TripleTranslationDataset(path=os.path.join(bpe_path, 'multi30k', 'val'),
                                        exts=exts, fields=fields,
                                        load_dataset=False, save_dataset=False)

    multi30k_it = data.BucketIterator(dev_data, 1024, device=device,
                                 batch_size_fn=batch_size_fn,
                                 train=False, repeat=False, shuffle=False,
                                 sort=False, sort_within_batch=True)
    args.__dict__.update({'FR': FR, "DE": DE, "EN": EN})
    return multi30k_it

iwslt_it = get_IWSLT(args.gpu, args.data_dir)
multi30k_it = get_multi30k(args, args.gpu, args.data_dir)
args.__dict__.update({'pad_token': 0,
                      'unk_token': 1,
                      'init_token': 2,
                      'eos_token': 3,
                      })

# Model
print(args)
print('Starting with HPARAMS: {}'.format(args.hp_str))
if args.gpu > -1 and torch.cuda.device_count() > 0:
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = lambda storage, loc: storage

print('CKPT {}'.format(args.ckpt))
model = get_model(args)
model.load_state_dict(torch.load(args.ckpt, map_location))


def iwslt_loop():
    with torch.no_grad():
        unbpe = True
        model.eval()
        en_hyp = []
        en_corpus = []

        # IWSLT loop
        print('IWSLT')
        for j, dev_batch in tqdm(enumerate(iwslt_it)):
            en_corpus.extend(args.EN.reverse(dev_batch.trg[0], unbpe=unbpe))
            fr, fr_len = dev_batch.src
            _, en_len = dev_batch.trg
            fr_hid = model.fr_en.enc(fr, fr_len)
            send_results = model.fr_en.dec.send(fr_hid, fr_len, en_len - 1, 'argmax')
            en = send_results["msg"]
            en_hyp.extend(args.EN.reverse(en, unbpe=True))
        return en_corpus, en_hyp


def multi30k_loop():
    with torch.no_grad():
        unbpe = True
        model.eval()
        en_hyp = []
        en_corpus = []

        # Multi30k loop
        print('Multi30k')
        for j, dev_batch in tqdm(enumerate(multi30k_it)):
            en_corpus.extend(args.EN.reverse(dev_batch.en[0], unbpe=unbpe))
            en, _ = model.fr_en_speak(dev_batch, is_training=False)
            en_hyp.extend(args.EN.reverse(en, unbpe=True))
    return en_corpus, en_hyp


multi30k_ref, multi30k_hyp = multi30k_loop()
multi30k_en_bleu = computeBLEU(multi30k_hyp, multi30k_ref, corpus=True)
#iwslt_ref, iwslt_hyp = iwslt_loop()
#iwslt_en_bleu = computeBLEU(iwslt_ref, iwslt_hyp, corpus=True)

print("multi30k bleu: {}".format(print_bleu(multi30k_en_bleu)))
#print("IWSLT bleu: {}".format(print_bleu(iwslt_en_bleu)))

with open(args.out, 'w') as f:
    f.write('\n'.join(multi30k_hyp))
