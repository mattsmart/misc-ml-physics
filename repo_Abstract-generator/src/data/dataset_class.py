import __init__

import io
import pandas as pd
import re
import requests
import string
from pathlib import Path, PurePath
from torch.utils.data import Dataset
from zipfile import ZipFile

import src.data.dataset_utils as utils
from src.settings import DIR_DATA, DIR_TOKENIZERS, VALID_TOKENIZATIONS,\
    SPECIAL_TOKEN_LIST


class BaseDataset(Dataset):
    """You can choose to set a transform (generally a tokenizer) to transform
    what is returned into another form of data more suitable (generally a
    tensor of tokens).

    To change what a sample is, need only change the method :
        get_instance_pretransform
    """
    def __init__(self, transform=None):
        self.category = 'base'
        self.name = ''
        self.data = []
        self.vocab_size = None
        self.transform = transform

        if self.transform is None:
            self.get_instance = self.get_instance_raw
        else:
            self.get_instance = self.get_instance_transformed

    def __len__(self):
        # gives number of samples in dataset
        return len(self.data)

    def __getitem__(self, idx):
        # uses function as defined by self.get_instance
        return self.get_instance(idx)

    def set_transform(self, transform):
        # can set transform and will change how we get an instance
        # Note: transform.encode could be used with rawTokenizer (instead of FastTokenizer)
        self.transform = transform
        self.get_instance = self.get_instance_transformed

    def get_instance_raw(self, idx):
        return self.data[idx]

    def get_instance_transformed(self, idx):
        """
        Once transform is defined, can get item already transformed
        Input
            idx (int) : index of sample to fetch and transform
        """
        instance = self.get_instance_raw(idx)
        instance = self.transform.encode(instance, return_tensors='pt')
        #  instance = self.transform.( instance, return_tensors='pt') # tokenize on-the-fly
        #  return instance['input_ids'][0]
        return instance[0]

    def tokenizer(self, flag_train,
                  tknzr_type='BPE',
                  vocab_size=None,
                  vocab=None,
                  flag_tknzr_fast=False,
                  tknzr_custom_name=None,
                  special_tokens=SPECIAL_TOKEN_LIST):

        # make token dir
        Path(DIR_TOKENIZERS).mkdir(parents=True, exist_ok=True)

        assert tknzr_type in VALID_TOKENIZATIONS
        if tknzr_custom_name is None:
            if self.name == '':
                name_file = f'{tknzr_type}_{self.category}.json'
            else:
                name_file = f'{tknzr_type}_{self.category}_{self.name}.json'
            tknzr_path = str(PurePath(DIR_TOKENIZERS, name_file))
        else:  # if a custom name was given, use that.
            tknzr_path = str(PurePath(DIR_TOKENIZERS, f'{tknzr_custom_name}'))

        if flag_train:  # training given the specific characteristics asked for
            tokenizer = utils.train_custom_tokenizer(
                self, tknzr_type, tknzr_path, vocab_size, vocab, flag_tknzr_fast,
                max_input_chars_per_word=None, **special_tokens)
        else:  # load from the custom name or default for the dataset chosen
            assert Path(tknzr_path).exists(), '''tokenizer does not exist, check
                                                file name. Can always choose
                                                flag_train=True to train your own'''
            tokenizer = utils.load_tokenizer(tknzr_path, flag_tknzr_fast, special_tokens)

        self.vocab_size = tokenizer.vocab_size
        self.set_transform(tokenizer)

        return tokenizer


class ArxivDataset(BaseDataset):
    """
    This Dataset takes the Arxiv data, downloads into a '.csv' files. When
    called, returns a sample from a list of samples with data_field.
    """
    def __init__(self, number_results=10 ** 4, search_query='all:electron',
                 transform=None, data_field='summary'):
        """
        Gets a list of arxiv paper metadata from a search_query. Downloads and
        saves into file if not already a file.

        Input
            nrb_results (int)           : number of samples of the query
            search_query (str)          : keywords that work as searchquery in arxiv_api
            data_field (str)            : name of the field used in training
            transform (function, opt)   : a transform of the data if one already exists
        """
        super().__init__(transform)
        self.category = 'arxiv'
        self.dataField = data_field
        # download data
        arxiv_dir = PurePath(DIR_DATA, self.category)
        Path(arxiv_dir).mkdir(parents=True, exist_ok=True)
        valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
        search_query_clean = re.sub(f'[^{valid_chars}]', "_", search_query)
        self.name = f'{search_query_clean}_{number_results}'

        file_path = PurePath(arxiv_dir, f'{self.name}.csv')
        if not Path(file_path).exists():
            utils.arxiv_api(file_path, max_results=number_results,
                            search_query=search_query)

        # make data, clean in a list
        self.data = pd.read_csv(file_path)
        # r'\s+|\\n' seems to be only one that works below
        remove_list = ['\r\n', '\n', '\ n', r'\\n', r'\n', r'\s+|\\n']
        self.data.replace(remove_list, ' ', regex=True, inplace=True)

    def get_instance_raw(self, idx):
        # returns some form of the text which will be our sample
        return self.data[self.dataField][idx]


class WikiTextDataset(BaseDataset):
    def __init__(self, data_name='wikitext-2-raw', bptt=35, transform=None):
        # only these work
        valid_wikitext = ['wikitext-2', 'wikitext-2-raw', 'wikitext-103', 'wikitext-103-raw']
        assert data_name in valid_wikitext
        assert isinstance(bptt, int)

        super().__init__(transform)
        self.category = data_name
        self.bptt = bptt

        # download data
        zipfile = f'{self.category}-v1.zip'
        url = f'https://s3.amazonaws.com/research.metamind.io/wikitext/{zipfile}'

        if not Path(DIR_DATA, self.category).exists():  # downloading zip file
            r = requests.get(url)                       # of wikitext
            z = ZipFile(io.BytesIO(r.content))
            z.extractall(DIR_DATA)

        split = ['wiki.train', 'wiki.valid', 'wiki.test']
        if 'raw' in self.category:
            split = [s + '.raw' for s in split]
        else:
            split = [s + '.raw' for s in split]

        # manipulate and make into list of samples
        for file in split:  # combine all train, test, valid text.
            data_string = ''
            with open(PurePath(DIR_DATA, self.category, file), 'r', encoding="utf-8") as f:
                data_string += ' '.join([line.strip() for line in f])
        word_list = data_string.split(' ')
        self.data = [' '.join(word_list[i:i + bptt]) for i in range(0, len(word_list), bptt)]


if __name__ == '__main__':
    dataset = ArxivDataset()
