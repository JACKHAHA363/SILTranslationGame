""" Our preprocess """
import os
from itertools import product
from os.path import join, basename
from subprocess import call
from torchtext.datasets import IWSLT
from tqdm import tqdm
import logging

ROOT_CORPUS_DIR = './data/corpus/'
ROOT_TOK_DIR = './data/tok'
ROOT_BPE_DIR = './data/bpe'
FR = '.fr'
EN = '.en'
DE = '.de'
MIN_FREQ = 10

LOGGER = logging.getLogger()


def config_logger():
    """ Config the logger """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    LOGGER.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    LOGGER.handlers = [console_handler]


config_logger()

"""
Download
"""


def _IWSLT_download_helper(src_lang, tgt_lang):
    """ Download result given source and target language """
    corpus_dir = join(ROOT_CORPUS_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(corpus_dir):
        LOGGER.info('iwslt {}-{} exists, skipping...'.format(src_lang[1:], tgt_lang[1:], corpus_dir))
        return
    LOGGER.info('downloading in {}...'.format(corpus_dir))
    IWSLT.dirname = IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:])
    IWSLT.urls = [IWSLT.base_url.format(src_lang[1:], tgt_lang[1:], IWSLT.dirname)]
    IWSLT.download(root=ROOT_CORPUS_DIR, check=corpus_dir)
    IWSLT.clean(corpus_dir)


_IWSLT_download_helper(FR, EN)
_IWSLT_download_helper(EN, DE)


def _download_multi30k():
    """ Get the corpus of multi30k task1 """
    corpus_dir = join(ROOT_CORPUS_DIR, 'multi30k')
    if os.path.exists(corpus_dir):
        LOGGER.info('multi30k exists, skipping...')
        return
    LOGGER.info('Downloading multi30k task1...')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    base_url = 'https://github.com/multi30k/dataset/raw/master/data/task1/raw/{}{}.gz'
    for prefix, lang in product(prefixs, langs):
        wget_cmd = ['wget', base_url.format(prefix, lang), '-P', corpus_dir]
        call(wget_cmd)
        call(['gunzip', '-k', join(corpus_dir, '{}{}.gz'.format(prefix, lang))])


_download_multi30k()

"""
Tokenize
"""
MOSES_PATH = os.path.join(os.path.dirname(__file__), 'moses/tokenizer')


def _tokenize(in_file, out_file, lang):
    cmd = ['cat', in_file, '|']
    cmd += [os.path.join(MOSES_PATH, 'lowercase.perl'), '|']
    cmd += [os.path.join(MOSES_PATH, 'normalize-punctuation.perl'), '-l', lang[1:], '|']
    cmd += [os.path.join(MOSES_PATH, 'tokenizer.perl'), '-l', lang[1:], '-threads', '4']
    cmd += ['>', out_file]
    cmd = ' '.join(cmd)
    LOGGER.info(cmd)
    os.system(cmd)


def _tokenize_IWSLT_helper(src_lang, tgt_lang):
    """ Tokenize one of the IWSLT """
    token_dir = join(ROOT_TOK_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(token_dir):
        LOGGER.info('{} exists, skipping...'.format(token_dir))
        return
    os.makedirs(token_dir)
    corpus_dir = join(ROOT_CORPUS_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))

    # Get all suffix
    suffix_langs = [(src_lang[1:] + '-' + tgt_lang[1:] + src_lang, src_lang),
                    (src_lang[1:] + '-' + tgt_lang[1:] + tgt_lang, tgt_lang)]

    # Get all prefix
    prefixs = ['train', 'IWSLT16.TED.tst2013']

    for prefix, (suffix, lang) in product(prefixs, suffix_langs):
        in_file = join(corpus_dir, prefix + '.' + suffix)
        out_file = join(token_dir, prefix + '.' + suffix)
        _tokenize(in_file=in_file, out_file=out_file, lang=lang)


_tokenize_IWSLT_helper(FR, EN)
_tokenize_IWSLT_helper(EN, DE)


def _tokenize_multi30k():
    # tokenize
    corpus_dir = join(ROOT_CORPUS_DIR, 'multi30k')
    prefixs = ['train', 'val', 'test_2017_flickr']
    langs = [FR, EN, DE]
    tok_dir = join(ROOT_TOK_DIR, 'multi30k')
    if os.path.exists(tok_dir):
        LOGGER.info('multi30k tokens exists, skipping...')
        return
    LOGGER.info('Tokenizing multi30k task1...')
    os.makedirs(tok_dir)
    for prefix, lang in product(prefixs, langs):
        file_name = '{}{}'.format(prefix, lang)
        in_file = join(corpus_dir, file_name)
        out_file = join(tok_dir, file_name)
        _tokenize(in_file, out_file, lang)


_tokenize_multi30k()

"""
LEARN BPE and apply it to corpus
"""

SUBWORD = join(os.path.dirname(__file__), 'subword-nmt')
LEARN_BPE = join(SUBWORD, 'learn_bpe.py')
APPLY_BPE = join(SUBWORD, 'apply_bpe.py')

def learn_bpe():
    """ Learn the BPE and get vocab """
    if not os.path.exists(ROOT_BPE_DIR):
        os.makedirs(ROOT_BPE_DIR)

    if not os.path.exists(join(ROOT_BPE_DIR, 'bpe.codes')):
        cmd = ['cat']
        cmd += [line.rstrip('\n') for line in os.popen('find ' + ROOT_TOK_DIR + ' -regex ".*\.en"')]
        cmd += [line.rstrip('\n') for line in os.popen('find ' + ROOT_TOK_DIR + ' -regex ".*\.fr"')]
        cmd += [line.rstrip('\n') for line in os.popen('find ' + ROOT_TOK_DIR + ' -regex ".*\.de"')]
        cmd += ['|']
        cmd += [LEARN_BPE, '-s', '25000']
        cmd += ['>', join(ROOT_BPE_DIR, 'bpe.codes')]
        cmd = ' '.join(cmd)
        LOGGER.info(cmd)
        os.system(cmd)
    else:
        LOGGER.info('bpe.codes file exist, skipping...')


learn_bpe()


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


def apply_bpe_iwslt(src_lang, tgt_lang):
    """ Apply BPE to iwslt with `src_lang` and `tgt_lang` """
    bpe_dir = join(ROOT_BPE_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    if os.path.exists(bpe_dir):
        LOGGER.info('BPE IWSLT for {}-{} exists, skipping...'.format(src_lang[1:], tgt_lang[1:]))
        return
    os.makedirs(bpe_dir)
    tok_dir = join(ROOT_TOK_DIR, IWSLT.name, IWSLT.base_dirname.format(src_lang[1:], tgt_lang[1:]))
    suffixs = [src_lang[1:] + '-' + tgt_lang[1:] + src_lang,
               src_lang[1:] + '-' + tgt_lang[1:] + tgt_lang]
    prefixs = ['train', 'IWSLT16.TED.tst2013']
    for prefix, suffix in product(prefixs, suffixs):
        tokenized_file = join(tok_dir, prefix + '.' + suffix)
        bpe_out = join(bpe_dir, prefix + '.' + suffix)
        _apply_bpe(in_file=tokenized_file, out_file=bpe_out)


def apply_bpe_multi30k():
    """ Apply BPE to multi30k """
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


apply_bpe_multi30k()
apply_bpe_iwslt(FR, EN)
apply_bpe_iwslt(EN, DE)

"""
Converting vocab to itos
"""
import torch
from collections import Counter

for lang in [FR, EN, DE]:
    counter = Counter()
    for fpath in os.popen('find ' + ROOT_BPE_DIR + ' -regex ".*\{}"'.format(lang)):
        fpath = fpath.rstrip('\n')
        with open(fpath) as f:
            for line in f:
                for word in line.rstrip('\n').split():
                    counter[word] += 1
    LOGGER.info('{} vocab size: {}'.format(lang, len(counter)))
    torch.save(counter, join(ROOT_BPE_DIR, 'vocab' + lang + '.pth'))
