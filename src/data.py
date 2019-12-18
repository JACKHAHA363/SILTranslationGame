import io
import six
import torch
import _pickle as pickle
import revtok
import os
from os.path import abspath, dirname, join

from torchtext import data, datasets, vocab
from torchtext.data import Dataset
from collections import Counter, OrderedDict, defaultdict
from contextlib import ExitStack

from utils import which_machine, cuda

def dyn_batch_without_padding(new, i, sofar):
    if hasattr(new, "src"):
        return sofar + max(len(new.src), len(new.trg))
    elif hasattr(new, "fr"):
        return sofar + max(len(new.fr), len(new.en), len(new.de))
    elif hasattr(new, "_1"):
        return sofar + max([len(new.__dict__[x]) for x in "_1 _2 _3 _4 _5".split()])
    else:
        raise Exception


batch_size_fn = dyn_batch_without_padding


def build_dataset(args, dataset):
    device = "cuda:{}".format(args.gpu) if args.gpu > -1 else "cpu"

    # Determine corpora path
    machine = which_machine()
    if machine == 'mila':
        vocab_path = join(dirname(dirname(dirname(abspath(__file__)))), 'data/bpe')
    else:
        raise ValueError
    en_vocab, de_vocab, fr_vocab = "vocab.en.pth", "vocab.de.pth", "vocab.fr.pth"
    train_repeat = False if args.setup == "ranker" else True
    if dataset == "iwslt":
        SRC = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                          include_lengths=True, batch_first=True)
        TRG = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                          include_lengths=True, batch_first=True)

        src, trg = ["."+xx for xx in args.pair.split("_")]
        pair = "en-de" if args.pair == "en_de" else "en-fr"

        vocabs = [en_vocab, de_vocab] if pair == "en-de" else [fr_vocab, en_vocab]

        for (field, vocab) in zip([SRC, TRG], vocabs):
            field.vocab = TextVocab(itos=torch.load(join(vocab_path, vocab)))

        train_path = join(vocab_path, 'iwslt', pair, 'train.' + pair)
        dev_path = join(vocab_path, 'iwslt', pair, 'IWSLT16.TED.tst2013.' + pair)
        train_data = NormalTranslationDataset(path=train_path,
                                              exts=(src, trg), fields=(SRC, TRG),
                                              load_dataset=args.load_dataset, save_dataset=args.save_dataset,
                                              training_max_len=args.training_max_len)

        dev_data = NormalTranslationDataset(path=dev_path,
                                            exts=(src, trg), fields=(SRC, TRG),
                                            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        train_it = data.BucketIterator(train_data, args.batch_size, device=device,
                                       batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True,
                                       sort=False, sort_within_batch=True)
        dev_it = data.BucketIterator(dev_data, args.batch_size, device=device,
                                     batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False,
                                     sort=False, sort_within_batch=True)

        args.__dict__.update({'src':SRC, "trg":TRG, 'voc_sz_src':len(SRC.vocab), 'voc_sz_trg':len(TRG.vocab)})

    elif dataset == "multi30k":
        FR   = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                           include_lengths=True, batch_first=True)
        EN   = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                           include_lengths=True, batch_first=True)
        DE   = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                           include_lengths=True, batch_first=True)

        vocabs = [fr_vocab, en_vocab, de_vocab]
        fields = [FR, EN, DE]
        exts = ['.fr', '.en', '.de']

        for (field, vocab) in zip(fields, vocabs):
            field.vocab = TextVocab(itos=torch.load(vocab_path + vocab))

        train = "train.bpe"
        dev = "val.bpe"

        train_data = TripleTranslationDataset(path=data_prefix + train,
            exts=exts, fields=fields,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        dev_data = TripleTranslationDataset(path=data_prefix + dev,
            exts=exts, fields=fields,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        train_it  = Multi30kIterator(train_data, args.batch_size, device=device, \
                                        batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True, \
                                        sort=False, sort_within_batch=True)
        dev_it    = Multi30kIterator(dev_data, args.batch_size, device=device, \
                                        batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False, \
                                        sort=False, sort_within_batch=True)

        if args.setup == "single":
            langs = {"en":EN, "de":DE, "fr":FR}
            src, trg = args.pair.split("_")
            src, trg = langs[src], langs[trg]
            args.__dict__.update({'voc_sz_src':len(src.vocab), 'voc_sz_trg':len(trg.vocab)})
            args.__dict__.update({'src':src, 'trg':trg})
        else:
            args.__dict__.update({'FR':FR, "DE":DE, "EN":EN})

    elif dataset == "coco":
        EN   = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                           include_lengths=True, batch_first=True)
        EN.vocab = TextVocab(itos=torch.load(vocab_path + en_vocab))

        fields = [EN]*5
        exts = ['.1', '.2', '.3', '.4', '.5']

        train = "train.bpe"
        dev = "valid.bpe"

        train_data = ParallelTranslationDataset(path=data_prefix + train,
            exts=exts, fields=fields,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        dev_data = ParallelTranslationDataset(path=data_prefix + dev,
            exts=exts, fields=fields,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        train_it  = Multi30kIterator(train_data, args.batch_size, device=device, \
                                        batch_size_fn=batch_size_fn, train=True, repeat=train_repeat, shuffle=True, \
                                        sort=False, sort_within_batch=True)
        dev_it    = Multi30kIterator(dev_data, args.batch_size, device=device, \
                                        batch_size_fn=batch_size_fn, train=False, repeat=False, shuffle=False, \
                                        sort=False, sort_within_batch=True)

        args.__dict__.update({"EN":EN})

    elif dataset in ['wikitext2', 'wikitext103']:
        EN   = NormalField(init_token="<BOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>", \
                          batch_first=False)
        EN.vocab = TextVocab(itos=torch.load(vocab_path + en_vocab))

        train = "wiki.train.raw.bpe"
        dev = "wiki.valid.raw.bpe"

        train_data = LanguageModelingDataset(path=data_prefix + train, field=EN,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        dev_data = LanguageModelingDataset(path=data_prefix + dev, field=EN,
            load_dataset=args.load_dataset, save_dataset=args.save_dataset)

        train_it  = data.BPTTIterator(train_data, args.batch_size, device=device, \
                                        train=True, repeat=train_repeat, shuffle=True,\
                                        bptt_len=args.seq_len)
        dev_it    = data.BPTTIterator(dev_data, args.batch_size, device=device, \
                                        train=False, repeat=False, \
                                        bptt_len=args.seq_len)

        args.__dict__.update({"EN":EN})

    args.__dict__.update({'pad_token': 0, \
                          'unk_token': 1, \
                          'init_token': 2, \
                          'eos_token': 3, \
                         })

    return train_it, dev_it

def _default_unk_index():
    return 1

"""
def data_path(dataset, args):

    if dataset == "multi30k":
        path="multi30k/data/task1/new"
    elif dataset == "coco":
        path="coco/captions"
    elif dataset == "wikitext103":
        path="wikitext/wikitext-103-raw"
    elif dataset == "wikitext2":
        path="wikitext/wikitext-2-raw"
    elif dataset == "iwslt":
        if args.pair == "fr_en":
            path="iwslt/en-fr"
        elif args.pair == "en_de":
            path="iwslt/en-de"
    else:
        path=""

    if machine == "nyu":
        return "/misc/kcgscratch1/ChoGroup/jason/corpora/{}/".format(path)
    elif machine == "fair":
        return "/private/home/jasonleeinf/corpora/{}/".format(path)
    else:
        return
"""

class TextVocab(vocab.Vocab):
    def __init__(self, counter=None, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None, itos=None):
        if itos is None:
            self.freqs = counter
            counter = counter.copy()
            min_freq = max(min_freq, 1)

            self.itos = list(specials)
            # frequencies of special tokens are not counted when building vocabulary
            # in frequency order
            for tok in specials:
                del counter[tok]

            max_size = None if max_size is None else max_size + len(self.itos)

            # sort by frequency, then alphabetically
            words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

            for word, freq in words_and_frequencies:
                if freq < min_freq or len(self.itos) == max_size:
                    break
                self.itos.append(word)
        else:
            self.itos = itos

        self.stoi = defaultdict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None


# load the dataset + reversible tokenization
class NormalField(data.Field):

    def reverse(self, batch, unbpe=True):
        if isinstance(batch, torch.Tensor):
            if not self.batch_first:
                batch = batch.t()
            with torch.cuda.device_of(batch):
                batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [ trim(ex, self.eos_token) for ex in batch] # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        if unbpe:
            batch = [ " ".join(filter(filter_special, ex)).replace("@@ ","") for ex in batch]
        else:
            batch = [ " ".join(filter(filter_special, ex) ) for ex in batch]
        return batch

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                counter.update(x)

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.pad_token, self.unk_token, self.init_token,
                            self.eos_token]
            if tok is not None))

        self.vocab = TextVocab(counter, specials=specials, **kwargs)

class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        if isinstance(ex, tuple):
            ex2 = ex[0]
        else:
            ex2 = ex
        if hasattr(ex2, "src"):
            return data.interleave_keys(len(ex2.src), len(ex2.trg))
        elif hasattr(ex2, "fr"):
            return data.interleave_keys(len(ex2.en), len(ex2.de))
        elif hasattr(ex2, "_1"):
            return data.interleave_keys(len(ex2._1), len(ex2._2))
        else:
            raise Exception

    def __init__(self, path, exts, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, path, exts, fields, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            root: Root dataset storage directory. Default is '.data'.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

class NormalTranslationDataset(TranslationDataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, exts, fields, load_dataset=False, save_dataset=False, training_max_len=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        dataset_path = path + exts[0] + exts[1] + '.processed.pt'
        if load_dataset and (os.path.exists(dataset_path)):
            examples = pickle.load(open(dataset_path, "rb"))
            print("Loaded TorchText dataset")
        else:
            print('Loading from {} and {}'.format(src_path, trg_path))
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        if training_max_len is None or ((len(src_line.split()) <= training_max_len)
                                                        and (len(trg_line.split()) <= training_max_len)):
                            examples.append(data.Example.fromlist(
                                [src_line, trg_line], fields))
            if save_dataset:
                pickle.dump(examples, open(dataset_path, "wb"))
                print("Saved TorchText dataset")

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class TripleTranslationDataset(TranslationDataset):

    def __init__(self, path, exts, fields, load_dataset=False, save_dataset=False, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('fr', fields[0]), ('en', fields[1]), ('de', fields[2])]

        src_path, trg_path, dec_path = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.triple.processed.pt')):
            examples = torch.load(path + '.triple.processed.pt')
        else:
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file, open(dec_path) as dec_file:
                for idx, (src_line, trg_line, dec_line) in enumerate(zip(src_file, trg_file, dec_file)):
                    src_line, trg_line, dec_line = src_line.strip(), trg_line.strip(), dec_line.strip()
                    if src_line != '' and trg_line != '' and dec_line != '':
                        examples.append(MyExample.fromlist(
                            [src_line, trg_line, dec_line], fields, idx))
            if save_dataset:
                torch.save(examples, open(path + '.triple.processed.pt', "wb"))
                print ("Saved TorchText dataset")

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class ParallelTranslationDataset(TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path, exts, fields, load_dataset=False, save_dataset=False, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [("_"+ext[1:], field) for (ext, field) in zip(exts, fields)]

        assert len(exts) == len(fields), 'N parallel dataset must match'
        self.N = len(fields)

        paths = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.multi.processed.pt')):
            examples = torch.load(path + '.multi.processed.pt')
        else:
            examples = []
            with ExitStack() as stack:
                files = [stack.enter_context(open(fname)) for fname in paths]
                for idx, lines in enumerate(zip(*files)):
                    lines = [line.strip() for line in lines]
                    if not any(line == '' for line in lines):
                        examples.append(MyExample.fromlist(lines, fields, idx))
            if save_dataset:
                torch.save(examples, path + '.multi.processed.pt')
                print ("Saved TorchText dataset")

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

class MyExample(data.Example):
    @classmethod
    def fromlist(cls, data, fields, idx):
        ex = cls()
        for (name, field), val in zip(fields, data):
            if field is not None:
                if isinstance(val, six.string_types):
                    val = val.rstrip('\n')
                # Handle field tuples
                if isinstance(name, tuple):
                    for n, f in zip(name, field):
                        setattr(ex, n, f.preprocess(val))
                else:
                    setattr(ex, name, field.preprocess(val))
        setattr(ex, "idx", idx)
        return ex

class Multi30kIterator(data.BucketIterator):
    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield Multi30kBatch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return

class Multi30kBatch(data.Batch):
    def __init__(self, data=None, dataset=None, device=None):
        if data is not None:
            self.batch_size = len(data)
            self.dataset = dataset
            self.fields = dataset.fields.keys()  # copy field names

            for (name, field) in dataset.fields.items():
                if field is not None:
                    batch = [getattr(x, name) for x in data]
                    setattr(self, name, field.process(batch, device=device))
            setattr(self, "idx", cuda( torch.LongTensor( [e.idx for e in data] ) ))

class LanguageModelingDataset(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, field, load_dataset=False, save_dataset=False, \
                 newline_eos=True, encoding='utf-8', **kwargs):
        fields = [('text', field)]
        if load_dataset and (os.path.exists(path + '.processed.pt')):
            examples = torch.load(path + '.processed.pt')
            print ("Loaded TorchText dataset")
        else:
            text = []
            with io.open(path, encoding=encoding) as f:
                for line in f:
                    if line.strip() != "":
                        text += field.preprocess(line.strip())
                        text.append('<EOS>')

            examples = [data.Example.fromlist([text], fields)]

            if save_dataset:
                torch.save(examples, open(path + '.processed.pt', "wb"))
                print ("Saved TorchText dataset")

        super(LanguageModelingDataset, self).__init__(
            examples, fields, **kwargs)
