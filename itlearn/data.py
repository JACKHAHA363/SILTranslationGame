import torch
import os
from os.path import join

from utils.data import UNK, BOS, EOS, batch_size_fn, TextVocab, NormalField, NormalTranslationDataset, \
    TripleTranslationDataset, ParallelTranslationDataset, LanguageModelingDataset, PAD
from utils.iterator import BucketIterator, Multi30kIterator, BPTTIterator


def get_s2p_dataset(args):
    its = {}
    device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"
    bpe_path = os.path.join(args.data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    for pair in ['fr_en', 'en_de']:
        small_dataset = hasattr(args, 's2p_small') and args.s2p_small
        _, _, train_it, _ = get_iwslt_iters(pair=pair, bpe_path=bpe_path,
                                            de_vocab=de_vocab, batch_size=args.batch_size,
                                            device=device, en_vocab=en_vocab,
                                            fr_vocab=fr_vocab, train_repeat=True,
                                            load_dataset=True, save_dataset=True,
                                            training_max_len=None, small_dataset=small_dataset)
        its[pair] = train_it
    return its


def build_dataset(args, dataset):
    device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"

    # Determine corpora path
    bpe_path = os.path.join(args.data_dir, 'bpe')
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    train_repeat = False if args.setup == "ranker" else True
    if 'iwslt' in dataset:
        small_dataset = dataset == 'iwslt_small'
        src_field, trg_field, train_it, dev_it = get_iwslt_iters(pair=args.pair, bpe_path=bpe_path,
                                                                 de_vocab=de_vocab, device=device,
                                                                 en_vocab=en_vocab, fr_vocab=fr_vocab,
                                                                 train_repeat=train_repeat,
                                                                 load_dataset=args.load_dataset,
                                                                 save_dataset=args.save_dataset,
                                                                 training_max_len=args.training_max_len,
                                                                 batch_size=args.batch_size,
                                                                 small_dataset=small_dataset)
        args.__dict__.update({'src': src_field, "trg": trg_field,
                              'voc_sz_src': len(src_field.vocab),
                              'voc_sz_trg': len(trg_field.vocab)})

    elif dataset == "multi30k":
        de, en, fr, train_it, dev_it = get_multi30k_iters(bpe_path=bpe_path, de_vocab=de_vocab, device=device,
                                                          en_vocab=en_vocab, fr_vocab=fr_vocab,
                                                          train_repeat=train_repeat,
                                                          load_dataset=args.load_dataset,
                                                          save_dataset=args.save_dataset,
                                                          batch_size=args.batch_size)

        if args.setup == "single":
            langs = {"en": en, "de": de, "fr": fr}
            src, trg = args.pair.split("_")
            src, trg = langs[src], langs[trg]
            args.__dict__.update({'voc_sz_src': len(src.vocab), 'voc_sz_trg': len(trg.vocab)})
            args.__dict__.update({'src': src, 'trg': trg})
        else:
            args.__dict__.update({'FR': fr, "DE": de, "EN": en})

    elif dataset == "coco":
        en, dev_it, train_it = get_coco_iters(bpe_path=bpe_path, device=device, en_vocab=en_vocab,
                                              train_repeat=train_repeat, batch_size=args.batch_size,
                                              load_dataset=args.load_dataset, save_dataset=args.save_dataset)
        args.__dict__.update({"EN": en})

    elif dataset in ['wikitext2', 'wikitext103']:
        en, dev_it, train_it = get_wikitext_iters(bpe_path=bpe_path, dataset=dataset, device=device,
                                                  en_vocab=en_vocab, train_repeat=train_repeat,
                                                  batch_size=args.batch_size, load_dataset=args.load_dataset,
                                                  save_dataset=args.save_dataset)
        args.__dict__.update({"EN": en})

    else:
        raise ValueError
    args.__dict__.update({'pad_token': 0,
                          'unk_token': 1,
                          'init_token': 2,
                          'eos_token': 3,
                         })

    return train_it, dev_it


def get_wikitext_iters(bpe_path, dataset, device, en_vocab, train_repeat, batch_size,
                       load_dataset, save_dataset, seq_len):
    EN = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     batch_first=False)
    EN.vocab = TextVocab(counter=torch.load(join(bpe_path, en_vocab)))
    train = join(bpe_path, dataset, 'wiki.train')
    dev = join(bpe_path, dataset, 'wiki.valid')
    train_data = LanguageModelingDataset(path=train, field=EN,
                                         load_dataset=load_dataset, save_dataset=save_dataset)
    dev_data = LanguageModelingDataset(path=dev, field=EN,
                                       load_dataset=load_dataset, save_dataset=save_dataset)
    train_it = BPTTIterator(train_data, batch_size, device=device,
                            train=True, repeat=train_repeat, shuffle=True,
                            bptt_len=seq_len)
    dev_it = BPTTIterator(dev_data, batch_size, device=device,
                          train=False, repeat=False,
                          bptt_len=seq_len)
    return EN, dev_it, train_it


def get_coco_iters(bpe_path, device, en_vocab, train_repeat, batch_size, load_dataset, save_dataset):
    en = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    en.vocab = TextVocab(counter=torch.load(join(bpe_path, en_vocab)))
    fields = [en] * 5
    exts = ['.1', '.2', '.3', '.4', '.5']
    train = "captions_train2014.bpe"
    dev = "captions_val2014.bpe"
    train_data = ParallelTranslationDataset(path=os.path.join(bpe_path, 'coco', train),
                                            exts=exts, fields=fields,
                                            load_dataset=load_dataset,
                                            save_dataset=save_dataset)
    dev_data = ParallelTranslationDataset(path=os.path.join(bpe_path, 'coco', dev),
                                          exts=exts, fields=fields,
                                          load_dataset=load_dataset,
                                          save_dataset=save_dataset)
    train_it = Multi30kIterator(train_data, batch_size, device=device,
                                batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True,
                                sort=False, sort_within_batch=True)
    dev_it = Multi30kIterator(dev_data, batch_size, device=device,
                              batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False,
                              sort=False, sort_within_batch=True)
    return en, dev_it, train_it


def get_multi30k_iters(bpe_path, de_vocab, device, en_vocab, fr_vocab, train_repeat,
                       load_dataset, save_dataset, batch_size):
    fr = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    en = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    de = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                     include_lengths=True, batch_first=True)
    vocabs = [fr_vocab, en_vocab, de_vocab]
    fields = [fr, en, de]
    exts = ['.fr', '.en', '.de']
    for (field, vocab) in zip(fields, vocabs):
        field.vocab = TextVocab(counter=torch.load(join(bpe_path, vocab)))
    train_data = TripleTranslationDataset(path=os.path.join(bpe_path, 'multi30k', 'train'),
                                          exts=exts, fields=fields,
                                          load_dataset=load_dataset, save_dataset=save_dataset)
    dev_data = TripleTranslationDataset(path=os.path.join(bpe_path, 'multi30k', 'val'),
                                        exts=exts, fields=fields,
                                        load_dataset=load_dataset, save_dataset=save_dataset)
    train_it = Multi30kIterator(train_data, batch_size, device=device,
                                batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True,
                                sort=False, sort_within_batch=True)
    dev_it = Multi30kIterator(dev_data, batch_size, device=device,
                              batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False,
                              sort=False, sort_within_batch=True)
    return de, en, fr, train_it, dev_it


def get_iwslt_iters(pair, bpe_path, de_vocab, device, en_vocab, fr_vocab, train_repeat, batch_size,
                    load_dataset=False, save_dataset=False, training_max_len=None, small_dataset=False):
    assert pair in ['fr_en', 'en_de']
    src_field = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                            include_lengths=True, batch_first=True)
    trg_field = NormalField(init_token=BOS, eos_token=EOS, pad_token=PAD, unk_token=UNK,
                            include_lengths=True, batch_first=True)
    src, trg = ["." + xx for xx in pair.split("_")]
    pair = pair.replace('_', '-')
    vocabs = [en_vocab, de_vocab] if pair == "en-de" else [fr_vocab, en_vocab]
    for (field, vocab) in zip([src_field, trg_field], vocabs):
        field.vocab = TextVocab(counter=torch.load(join(bpe_path, vocab)))
    train_path = join(bpe_path, 'iwslt', pair, 'train.' + pair)
    dev_path = join(bpe_path, 'iwslt', pair, 'IWSLT16.TED.tst2013.' + pair)
    train_data = NormalTranslationDataset(path=train_path,
                                          exts=(src, trg), fields=(src_field, trg_field),
                                          load_dataset=load_dataset, save_dataset=save_dataset,
                                          training_max_len=training_max_len)
    if small_dataset:
        print('nb examples: {} -> {}'.format(len(train_data), int(len(train_data) / 10)))
        train_data.examples = train_data.examples[:int(len(train_data) / 10)]

    dev_data = NormalTranslationDataset(path=dev_path,
                                        exts=(src, trg), fields=(src_field, trg_field),
                                        load_dataset=load_dataset, save_dataset=save_dataset)
    train_it = BucketIterator(train_data, batch_size, device=device,
                              batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True,
                              sort=False, sort_within_batch=True)
    dev_it = BucketIterator(dev_data, batch_size, device=device,
                            batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False,
                            sort=False, sort_within_batch=True)
    return src_field, trg_field, train_it, dev_it

