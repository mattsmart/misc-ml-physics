import io, requests, string, re
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from pathlib import Path, PurePath
from zipfile import ZipFile
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from src.default import RAW_DATA_DIR, TOK_DIR, SPECIAL_TOKEN_LIST
import src.data.dataset_utils as utils

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
        self.vocabSize = None
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
        self.transform = transform # transform.encode could be used with rawTokenizer (instead of FastTokenizer)
        self.get_instance = self.get_instance_transformed

    def get_instance_raw(self, idx):
        return self.data[idx]

    def get_instance_transformed(self, idx):
        """
        Once tranform is defined, can get item already transformed
        Input
            idx (int) : index of sample to fetch and transform
        """
        instance = self.get_instance_raw(idx)
        #instance = self.transform.( instance, return_tensors='pt') # tokenize on-the-fly
        #return instance['input_ids'][0]
        instance = self.transform.encode( instance, return_tensors='pt')
        return instance[0]

    def tokenizer(self, train, tknzrType=True, vocabSize=None, vocab=None,
                    tknzrFast=False, tknzrCustomName=None,
                    specialTokens=SPECIAL_TOKEN_LIST):

        # make token dir
        Path(TOK_DIR).mkdir(parents=True, exist_ok=True)

        if tknzrCustomName is None:
            if self.name == '': nameFile = f'{tknzrType}_{self.category}.json'
            else: nameFile = f'{tknzrType}_{self.category}_{self.name}.json'
            
            tknzrPath = str(PurePath(TOK_DIR, nameFile))
        else: # if a custom name was given, use that.
            tknzrPath = str(PurePath(TOK_DIR,f'{tknzrCustomName}'))

        if train: # training given the specific characteristics asked for
            tokenizer = utils.train_custom_tokenizer(self, tknzrType, tknzrPath,
                                                vocabSize, vocab, tknzrFast,
                                                max_input_chars_per_word=None,
                                                **specialTokens)
        else: # load from the custom name or default for the dataset chosen
            assert Path(tknzrPath).exists(), '''tokenizer does not exist, check
                                                file name. Can always choose
                                                train=True to train your own'''
            tokenizer = utils.load_tokenizer(tknzrPath, tknzrFast,
                                                specialTokens)

        self.vocabSize = tokenizer.vocab_size
        self.set_transform(tokenizer)

        return tokenizer

class ArxivDataset(BaseDataset):
    """
    This Dataset takes the Arxiv data, downloads into a '.csv' files. When
    called, returns a sample from a list of samples with data_field.
    """
    def __init__(self, nbrResults=10**4, searchQuery='all:electron',
                        transform=None, dataField='summary'):
        """
        Gets a list of arxiv paper metadata from a searchQuery. Downloads and
        saves into file if not already a file.

        Input
            nrbResults (int)            : number of samples of the query
            searchQuery (str)           : keywords that work as searchquery in
                                            arxiv_api
            dataField (str)             : name of the field used in training
            transform (function, opt)   : a transform of the data if one already
                                            exists
        """
        super().__init__(transform)
        self.category = 'arxiv'
        self.dataField = dataField
        # download data
        arxivDir = PurePath(RAW_DATA_DIR,self.category)
        Path(arxivDir).mkdir(parents=True, exist_ok=True)
        validChars = "-_.()%s%s" % (string.ascii_letters, string.digits)
        searchQueryClean = re.sub(f'[^{validChars}]', "_", searchQuery)
        self.name = f'{searchQueryClean}_{nbrResults}'
        filePath = PurePath(arxivDir,f'{self.name}.csv')
        if not Path(filePath).exists():
            utils.arxiv_api(filePath, nbrResults, searchQuery)

        # make data, clean in a list
        self.data = pd.read_csv(filePath)
        removeList = ['\r\n','\n','\ n',r'\\n',r'\n',r'\s+|\\n']   # r'\s+|\\n' seems to be
        self.data.replace(removeList,' ',regex=True, inplace=True) # the only one that works

    def get_instance_raw(self, idx):
        # returns some form of the text which will be our sample
        return self.data[self.dataField][idx]

class WikiTextDataset(BaseDataset):
    def __init__(self, dataName='wikitext-2-raw', bptt=35, transform=None):
        # only these work
        list = ['wikitext-2','wikitext-2-raw','wikitext-103','wikitext-103-raw']
        assert dataName in list
        assert isinstance(bptt, int)

        super().__init__(transform)
        self.category = dataName
        self.bptt = bptt

        # download data
        zipfile = f'{self.category}-v1.zip'
        url = f'https://s3.amazonaws.com/research.metamind.io/wikitext/{zipfile}'

        if not Path(RAW_DATA_DIR,self.category).exists(): # downloading zip file
            r = requests.get(url)                     # of wikitext
            z = ZipFile(io.BytesIO(r.content))
            z.extractall(RAW_DATA_DIR)

        split = ['wiki.train','wiki.valid','wiki.test']
        if 'raw' in self.category:
            split = [ s+'.raw' for s in split ]
        else:
            split = [ s+'.raw' for s in split ]

        # manipulate and make into list of samples
        for file in split: # combine all train, test, valid text.
            dataString = ''
            with open(PurePath(RAW_DATA_DIR,self.category,file), 'r') as f:
                dataString += ' '.join([line.strip() for line in f])
        wordList = dataString.split(' ')

        self.data = [' '.join(wordList[i:i + bptt]) for i in\
                                    range(0, len(wordList), bptt)]
