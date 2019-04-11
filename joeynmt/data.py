# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
import librosa
import torch
from typing import Optional

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


def make_data_iter(dataset: Dataset, batch_size: int, train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.Iterator(
            repeat=False, dataset=dataset, batch_size=batch_size,
            train=False, sort=False)

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        src_path = os.path.expanduser(path + ext)

        examples = []
        with open(src_path) as src_file:
            for src_line in src_file:
                src_line = src_line.strip()
                if src_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line], fields))

        super(MonoDataset, self).__init__(examples, fields, **kwargs)


def load_audio_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                    Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).
    
    The training data is filtered to include sentences up to `max_sent_length`
    on text side and audios up to `max_audio_length`.
    
    :param cfg: configuration dictionary for data
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: copy of trg_vocab
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    data_cfg = cfg["data"]
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    if data_cfg["audio"] == "src":
        audio_lang = src_lang
    else:
        audio_lang = trg_lang
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg.get("max_sent_length", sys.maxsize)
    max_audio_length = data_cfg.get("max_audio_length", sys.maxsize)
    mfcc_number = cfg["model"]["encoder"]["embeddings"]["embedding_dim"]
    assert mfcc_number <= 80,\
    "The number of used MFCCs could not be higher than the number of Mel bands. Change the encoder's embedding_dim."
    check_ratio = data_cfg.get("input_length_ratio", sys.maxsize)

    #pylint: disable=unnecessary-lambda
    if level == "char":
        tok_fun = lambda s: list(s)
        char = True
    else:  # bpe or word, pre-tokenized
        tok_fun = lambda s: s.split()
        char = False

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = AudioDataset(path=train_path, text_ext="." + audio_lang,
                              audio_ext=".txt", sfield=src_field, tfield=trg_field, 
                              num=mfcc_number, char_level=char, train=True, 
                              check=check_ratio, filter_pred = lambda x: 
                              len(vars(x)['src']) <= max_audio_length
                              and len(vars(x)['trg']) <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    trg_vocab_file = data_cfg.get(audio_lang + "_vocab", None)
    src_vocab_file = None
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq, max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq, max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)

    dev_data = AudioDataset(path=dev_path, text_ext="." + audio_lang, audio_ext=".txt", 
                            sfield=src_field, tfield=trg_field, num=mfcc_number,
                            char_level=char, train=False, check=check_ratio)
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + audio_lang):
            test_data = AudioDataset(path=test_path, text_ext="." + audio_lang, 
                            audio_ext=".txt", sfield=src_field, tfield=trg_field, num=mfcc_number, 
                            char_level=char, train=False, check=check_ratio)
        else:
            # no target is given -> create dataset from src only
            test_data = MonoAudioDataset(path=test_path, audio_ext=".txt", 
                            field=trg_field, num=mfcc_number, char_level=char)
    trg_field.vocab = trg_vocab
    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


class AudioDataset(TranslationDataset):
    """Defines a dataset for speech recognition/translation."""

    def __init__(self, path: str, text_ext: str, audio_ext: str, sfield: Field, tfield: Field, 
                 num: int, char_level: bool, train: bool, check: int, **kwargs) -> None:
        """Create an AudioDataset given path and fields.

            :param path: Prefix of path to the data files
            :param text_ext: Containing the extension to path for text file
            :param audio_ext: Containing the extension to path for audio file
            :param fields: Containing the fields that will be used for text data
            :param num: Containing the number of mfccs to extract (= dimension of source embeddings)
            :param char_level: Containing the indicator for char level
            :param train: Containing the indicator for training set 
            :param check: Containing the length ratio as a filter for training set
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        all_fields = [('trg', tfield), ('mfcc', audio_field), ('src', sfield)]

        text_path = os.path.expanduser(path + text_ext)
        audio_path = os.path.expanduser(path + audio_ext)
        examples = []
        if train :
            maxi = 1
            mini = 10
            summa = 0
            count = 0
            log_path = os.path.expanduser(path + '_length_statistics')
            length_info = open(log_path, 'a')

        if len(open(text_path).read().splitlines()) != len(open(audio_path).read().splitlines()):
            raise IndexError('The size of the text and audio dataset differs.')
        else:
            with open(text_path) as text_file, open(audio_path) as audio_file:
                for text_line, audio_line in zip(text_file, audio_file):
                    text_line = text_line.strip()
                    audio_line = audio_line.strip()
                    y, sr = librosa.load(audio_line, sr=None)
                    # overwrite default values for the window width of 25 ms and stride of 10 ms (for sr = 16kHz)
                    # (n_fft : length of the FFT window, hop_length : number of samples between successive frames)
                    # default values: n_fft=2048, hop_length=512, n_mels=128
                    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num, n_fft=int(sr/40), hop_length=int(sr/100), n_mels=80)
                    featuresT = features.T
                    featureS = torch.Tensor(featuresT)
                    if char_level :
                        audio_dummy = "a" * (featuresT.shape[0] - 1) #generate a line with <unk> of given size
                    else :
                        audio_dummy = "a " * (featuresT.shape[0] - 1) #generate a line with <unk> of given size
                    if train :
                        length_ratio = featuresT.shape[0] // (len(text_line) + 1)
                        if text_line != '' and audio_line != '' and os.path.getsize(audio_line) > 44  and length_ratio < check :
                            examples.append(data.Example.fromlist([text_line, featureS, audio_dummy], all_fields))
                        if length_ratio > maxi:
                            maxi = length_ratio
                        if length_ratio < mini:
                            mini = length_ratio
                        summa += length_ratio
                        count += 1
                    else:
                        if text_line != '' and audio_line != '' and os.path.getsize(audio_line) > 44 :
                            examples.append(data.Example.fromlist([text_line, featureS, audio_dummy], all_fields))
        if train :
            length_info.write('mini={0}, maxi={1}, mean={2}, checked by {3} \n'.format(mini, maxi, summa/count, check))
            length_info.close()
        super(TranslationDataset, self).__init__(examples, all_fields, **kwargs)

    def __len__(self):
        return len(self.examples)

    def gettext(self, index):
        return self.examples[index].trg

    def getaudio(self, index):
        return self.examples[index].audio


class MonoAudioDataset(TranslationDataset):
    """Defines a dataset for speech recognition/translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, audio_ext: str, field: Field, num: int, char_level: bool, **kwargs) -> None:
        """
        Create a MonoAudioDataset (=only sources) given path.

            :param path: Prefix of path to the data file
            :param audio_ext: Containing the extension to path for audio file
            :param field: Containing the field for dummy audio data
            :param num: Containing the number of mfccs to extract (= dimension of source embeddings)
            :param char_level: Containing the indicator for char level
            :param kwargs: Passed to the constructor of data.Dataset.
        """
        audio_field = data.RawField()
        fields = [('mfcc', audio_field), ('src', field)]
        audio_path = os.path.expanduser(path + audio_ext)
        examples = []

        with open(audio_path) as audio_file:
            for audio_line in audio_file:
                audio_line = audio_line.strip()
                y, sr = librosa.load(audio_line, sr=None)
                features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num, n_fft=int(sr/40), hop_length=int(sr/100), n_mels=80)
                featuresT = features.T
                featureS = torch.Tensor(featuresT)
                if char_level :
                    audio_dummy = "a" * (featuresT.shape[0] - 2) #generate a line with <unk> of given size
                else :
                    audio_dummy = "a " * (featuresT.shape[0] - 2) #generate a line with <unk> of given size
                if audio_line != '' and os.path.getsize(audio_line) > 44 :
                    examples.append(data.Example.fromlist([featureS, audio_dummy], fields))
        super(TranslationDataset, self).__init__(examples, fields, **kwargs)

    def __len__(self):
        return len(self.examples)
