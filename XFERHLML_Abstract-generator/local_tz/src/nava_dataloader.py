import urllib.request
import feedparser
import pandas as pd
import spacy
import numpy as np
from collections import defaultdict

# Base api query url
base_url = 'http://export.arxiv.org/api/query?';

# Search parameters
search_query = 'all:electron' # search for electron in all fields
start = 0                     # retreive the first 5 results
#max_results = 10**4
max_results = 10

query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                     start,
                                                     max_results)

# perform a GET request using the base_url and query
response = urllib.request.urlopen(base_url+query).read()

# parse the response using feedparser
feed = feedparser.parse(response)

#coloums of interasts (maybe?)
col=['title', 'summary', 'authors', 'arxiv_primary_category', 'tags']

# Run through each entry, and fill the information into a list
data_list=[]
for c in col:
	abstract_list=[]
	for entry in feed.entries:
		abstract_list.append(entry.get(c))
	data_list.append(abstract_list)

# convert into a panda dataframe (maybe more visible + have some pros I might need)
data_df = pd.DataFrame(data_list,index=col)
data_df=data_df.T

# nlp = en_core_web_lg.load()
nlp = spacy.load("en_core_web_sm")

#taking just the titles (data_list[0]). (maybe to use the summary instaed?)
# using lower case. removing extra spaces and '\n ' 
doc=[nlp.tokenizer(text.lower().replace('\n ','').strip()) for text in data_list[0]]

class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}#{PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 0
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in [token.text for token in sentence]:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

voc=Vocabulary('test')
for sent in doc:
 	voc.add_sentence(sent)

Input_list=[]
for sample in range(len(doc)):
#  Input_list.append([token.rank for token in doc[sample]])
	Input_list.append([voc.to_index(token.text) for token in doc[sample]])
Output_list=Input_list;
Input_Output_Data_list=[Input_list,Output_list]

## next: to understand / complete
from sklearn.model_selection import train_test_split

#10% test set
In_train, In_test, Out_train, Out_test = train_test_split(Input_list, Output_list, test_size=0.1, random_state=1)

#from 90% train set --> 20% validation and 80 % training (= in total we have 10% test, 18% val, 72% train )
In_train, In_val, Out_train, Out_val = train_test_split(In_train, Out_train , test_size=0.2, random_state=1)

train_list=  In_train
label=Out_train
validation_list=In_val
test_list=In_test

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    label_list, text_list = [], []
    for _sample in batch:
        label_list.append(torch.tensor(_sample))
        text_list.append(torch.tensor(_sample))
    return pad_sequence(label_list, padding_value=0.0), pad_sequence(text_list, padding_value=0.0)

batch_size = 30

def create_iterators(batch_size=batch_size):
    """Heler function to create the iterators"""
    dataloaders = []
    for split in [train_list, validation_list, test_list]:
        dataloader = DataLoader(
            split, batch_size=batch_size,
            collate_fn=collate_batch
            )
        dataloaders.append(dataloader)
    return dataloaders

train_iterator, valid_iterator, test_iterator = create_iterators()

next(iter(train_iterator))