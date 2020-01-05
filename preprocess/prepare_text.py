""" Our preprocess """
import os
from itertools import product
from os.path import join, basename
from subprocess import call
from torchtext.datasets import IWSLT
import logging
import argparse
from shutil import rmtree
import json
import torch
import tqdm
from collections import Counter


def get_data_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True)
    args = parser.parse_args()
    return os.path.abspath(args.data_dir)


DATA_DIR = get_data_dir()
ROOT_TMP_DIR = os.path.join(os.path.dirname(DATA_DIR), 'tmp')
ROOT_CORPUS_DIR = os.path.join(ROOT_TMP_DIR, 'corpus')
ROOT_TOK_DIR = os.path.join(ROOT_TMP_DIR, 'tok')
ROOT_BPE_DIR = os.path.join(DATA_DIR, 'bpe')
FR = '.fr'
EN = '.en'
DE = '.de'
MOSES_PATH = os.path.join(os.path.dirname(__file__), 'moses/tokenizer')
SUBWORD = join(os.path.dirname(__file__), 'subword-nmt')
LEARN_BPE = join(SUBWORD, 'learn_bpe.py')
APPLY_BPE = join(SUBWORD, 'apply_bpe.py')
LOGGER = logging.getLogger()
log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
LOGGER.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
LOGGER.handlers = [console_handler]


def _tokenize(in_file, out_file, lang):
    cmd = ['cat', in_file, '|']
    cmd += [os.path.join(MOSES_PATH, 'lowercase.perl'), '|']
    cmd += [os.path.join(MOSES_PATH, 'normalize-punctuation.perl'), '-l', lang[1:], '|']
    cmd += [os.path.join(MOSES_PATH, 'tokenizer.perl'), '-l', lang[1:], '-threads', '4']
    cmd += ['>', out_file]
    cmd = ' '.join(cmd)
    LOGGER.info(cmd)
    os.system(cmd)


def _apply_bpe(in_file, out_file):
    """ Apply BPE """
    codes_file = join(ROOT_BPE_DIR, 'bpe.codes')
    assert os.path.exists(codes_file), '{} not exists!'.format(codes_file)
    cmd = [APPLY_BPE, '-c', codes_file]
    cmd += ['--input', in_file]
    cmd += ['--output', out_file]
    LOGGER.info('Applying BPE to {}'.format(basename(out_file)))
    LOGGER.info(' '.join(cmd))
    call(cmd)


def multi30k():
    if os.path.exists(join(ROOT_BPE_DIR, 'multi30k')) and os.path.exists(join(ROOT_BPE_DIR, 'bpe.codes')):
        LOGGER.info('Multi30K exists, skipping...')
        return

    # Download
    corpus_dir = join(ROOT_CORPUS_DIR, 'multi30k')
    LOGGER.info('Downloading multi30k task1...')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    base_url = 'https://github.com/multi30k/dataset/raw/master/data/task1/raw/{}{}.gz'
    for prefix, lang in product(prefixs, langs):
        wget_cmd = ['wget', base_url.format(prefix, lang), '-P', corpus_dir]
        call(wget_cmd)
        call(['gunzip', '-k', join(corpus_dir, '{}{}.gz'.format(prefix, lang))])
        call(['rm', join(corpus_dir, '{}{}.gz'.format(prefix, lang))])

    # Tokenize
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    tok_dir = join(ROOT_TOK_DIR, 'multi30k')
    LOGGER.info('Tokenizing multi30k task1...')
    os.makedirs(tok_dir)
    for prefix, lang in product(prefixs, langs):
        file_name = '{}{}'.format(prefix, lang)
        in_file = join(corpus_dir, file_name)
        out_file = join(tok_dir, file_name)
        _tokenize(in_file, out_file, lang)

    # Learn BPE
    if not os.path.exists(ROOT_BPE_DIR):
        os.makedirs(ROOT_BPE_DIR)
    cmd = ['cat']
    cmd += [line.rstrip('\n') for line in os.popen('find ' + join(ROOT_TOK_DIR, 'multi30k') + ' -regex ".*\.en"')]
    cmd += [line.rstrip('\n') for line in os.popen('find ' + join(ROOT_TOK_DIR, 'multi30k') + ' -regex ".*\.fr"')]
    cmd += [line.rstrip('\n') for line in os.popen('find ' + join(ROOT_TOK_DIR, 'multi30k') + ' -regex ".*\.de"')]
    cmd += ['|']
    cmd += [LEARN_BPE, '-s', '25000']
    cmd += ['>', join(ROOT_BPE_DIR, 'bpe.codes')]
    cmd = ' '.join(cmd)
    LOGGER.info(cmd)
    os.system(cmd)

    # Apply BPE
    bpe_dir = join(ROOT_BPE_DIR, 'multi30k')
    if os.path.exists(bpe_dir):
        LOGGER.info('BPE Multi30k exists, skipping...')
        return
    os.makedirs(bpe_dir)
    tok_dir = join(ROOT_TOK_DIR, 'multi30k')
    prefixs = ['train', 'val']
    langs = [FR, EN, DE]
    for prefix, lang in product(prefixs, langs):
        file_name = prefix + lang
        in_file = join(tok_dir, file_name)
        out_file = join(bpe_dir, file_name)
        _apply_bpe(in_file, out_file)


def iwslt():
    for src_lang, tgt_lang in zip([FR, EN], [EN, DE]):
        bpe_dir = join(ROOT_BPE_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
        if os.path.exists(bpe_dir):
            LOGGER.info('BPE IWSLT for {}-{} exists, skipping...'.format(src_lang[1:], tgt_lang[1:]))
            continue
        os.makedirs(bpe_dir)

        # Download
        corpus_dir = join(ROOT_CORPUS_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
        LOGGER.info('downloading in {}...'.format(corpus_dir))
        IWSLT.dirname = IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:])
        IWSLT.urls = [IWSLT.base_url.format(src_lang[1:], tgt_lang[1:], IWSLT.dirname)]
        IWSLT.download(root=ROOT_CORPUS_DIR, check=corpus_dir)
        IWSLT.clean(corpus_dir)

        # Tokenize
        token_dir = join(ROOT_TOK_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
        os.makedirs(token_dir)
        suffix_langs = [(src_lang[1:] + '-' + tgt_lang[1:] + src_lang, src_lang),
                        (src_lang[1:] + '-' + tgt_lang[1:] + tgt_lang, tgt_lang)]
        prefixs = ['train', 'IWSLT16.TED.tst2013']
        for prefix, (suffix, lang) in product(prefixs, suffix_langs):
            in_file = join(corpus_dir, prefix + '.' + suffix)
            tok_file = join(token_dir, prefix + '.' + suffix)
            bpe_file = join(bpe_dir, prefix + '.' + suffix)
            _tokenize(in_file=in_file, out_file=tok_file, lang=lang)
            _apply_bpe(in_file=tok_file, out_file=bpe_file)


def coco():
    """ Download coco captions """
    if os.path.exists(join(ROOT_BPE_DIR, 'coco')):
        LOGGER.info('COCO captions exists, skipping...')
        return
    os.makedirs(join(ROOT_BPE_DIR, 'coco'))
    
    # Download
    corpus_dir = join(ROOT_CORPUS_DIR, 'coco')
    wget_cmd = ['wget', "http://images.cocodataset.org/annotations/annotations_trainval2014.zip", '-P', corpus_dir]
    call(wget_cmd)
    unzip_cmd = ['unzip', join(corpus_dir, 'annotations_trainval2014.zip'), '-d', corpus_dir]
    call(unzip_cmd)
    call(['rm', join(corpus_dir, 'annotations_trainval2014.zip')])

    # Extract from json
    for name in ['captions_val2014', 'captions_train2014']:
        json_file = join(corpus_dir, 'annotations', name + '.json')
        json_str = open(json_file).read()
        decode = json.loads(json_str)
        annotations = decode['annotations']
        captions = {}
        for annotation in tqdm.tqdm(annotations):
            caption = annotation['caption'].rstrip('\n')
            if caption == "":
                raise ValueError
            if annotation['image_id'] in captions:
                captions[annotation['image_id']].append(caption)
            else:
                captions[annotation['image_id']] = [caption]
        for i in range(1, 6):
            orig_file = join(corpus_dir, name + '.txt.' + str(i))
            lines = [val[i - 1] for val in captions.values()]
            with open(orig_file, 'w') as f:
                f.write('\n'.join(lines))

            # Tokenize
            tok_file = join(corpus_dir, name + '.tok.' + str(i))
            _tokenize(in_file=orig_file, out_file=tok_file, lang=EN)

            # BPE
            bpe_file = join(ROOT_BPE_DIR, 'coco', name + '.bpe.' + str(i))
            _apply_bpe(tok_file, bpe_file)


def get_vocab():
    for lang in [FR, EN, DE]:
        if not os.path.exists(join(ROOT_BPE_DIR, 'vocab' + lang + '.pth')):
            counter = Counter()
            for fpath in os.popen('find ' + join(ROOT_BPE_DIR, 'iwslt/fr-en') + ' -regex ".*\{}"'.format(lang)):
                fpath = fpath.rstrip('\n')
                with open(fpath) as f:
                    for line in f:
                        for word in line.rstrip('\n').split():
                            counter[word] += 1
            for fpath in os.popen('find ' + join(ROOT_BPE_DIR, 'multi30k') + ' -regex ".*\{}"'.format(lang)):
                fpath = fpath.rstrip('\n')
                with open(fpath) as f:
                    for line in f:
                        for word in line.rstrip('\n').split():
                            counter[word] += 1
            LOGGER.info('{} vocab size: {}'.format(lang, len(counter)))
            torch.save(counter, join(ROOT_BPE_DIR, 'vocab' + lang + '.pth'))
        else:
            LOGGER.info('vocab {} exists, skipping...'.format(lang[1:]))


def flickr30k_caps():
    flickr30k_dir = os.path.join(DATA_DIR, 'flickr30k')
    if not os.path.exists(join(flickr30k_dir, 'train.txt')):
        call(['wget', 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/train.txt', '-P',
              join(flickr30k_dir)])
        call(['wget', 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/image_splits/val.txt', '-P',
              join(flickr30k_dir)])

    cap_dir = os.path.join(flickr30k_dir, 'caps')
    raw_url = "https://github.com/multi30k/dataset/raw/master/data/task2/raw/{}.{}.en.gz"
    if not os.path.exists(cap_dir):
        # Download original
        raw_dir = os.path.join(cap_dir, 'raw')
        for split in ['train', 'val']:
            for i in range(1, 6):
                wget_cmd = ['wget', raw_url.format(split, i), '-P', raw_dir]
                call(wget_cmd)
                call(['gunzip', '-k', join(raw_dir, '{}.{}.en.gz'.format(split, i))])
                call(['rm', join(raw_dir, '{}.{}.en.gz'.format(split, i))])

                # Tokenize
                infile = join(raw_dir, '{}.{}.en'.format(split, i))
                outfile = join(raw_dir, '{}.{}.en.tok'.format(split, i))
                _tokenize(infile, outfile, EN)

                # BPE
                bpe_out = join(cap_dir, '{}.{}.bpe'.format(split, i))
                _apply_bpe(outfile, bpe_out)

        # Remove corpus
        rmtree(raw_dir)


if __name__ == '__main__':
    multi30k()
    iwslt()
    flickr30k_caps()
    coco()
    get_vocab()
    #LOGGER.info('Cleaning...')
    #rmtree(ROOT_TMP_DIR)
