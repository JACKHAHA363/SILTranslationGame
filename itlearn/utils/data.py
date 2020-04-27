import io
import os
import pickle as pickle
from collections import Counter, OrderedDict
from contextlib import ExitStack

import six
import torch
from torchtext import vocab, data
from torchtext.data import Dataset

from utils.misc import cuda

UNK = '<unk>'
BOS = '<bos>'
EOS = '<eos>'
PAD = '<pad>'


def trim_batch(batch, ratio):
    if hasattr(batch, "fr"):
        total_size = batch.fr[0].shape[0]
        split = int(ratio * total_size) + 1
        batch.fr[0] = batch.fr[0][:split]
        batch.fr[1] = batch.fr[1][:split]

        batch.en[0] = batch.en[0][:split]
        batch.en[1] = batch.en[1][:split]

        batch.de[0] = batch.de[0][:split]
        batch.de[1] = batch.de[1][:split]
    elif hasattr(batch, "src"):
        total_size = batch.src[0].shape
        split = int(ratio * total_size) + 1
        batch.src[0] = batch.src[0][:split]
        batch.src[1] = batch.src[1][:split]

        batch.trg[0] = batch.trg[0][:split]
        batch.trg[1] = batch.trg[1][:split]
    else:
        raise Exception
    return batch


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


class TextVocab(vocab.Vocab):
    def __init__(self, counter):
        super(TextVocab, self).__init__(counter=counter,
                                        specials=[PAD, UNK, BOS, EOS],
                                        specials_first=True)


class NormalField(data.Field):

    def reverse(self, batch, unbpe=True, remove_dots=False):
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

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos

        specials = [self.init_token, self.pad_token]
        if remove_dots:
            specials.append('.')

        def filter_special(tok):
            return tok not in specials

        if unbpe:
            batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
        else:
            batch = [" ".join(filter(filter_special, ex) ) for ex in batch]
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
