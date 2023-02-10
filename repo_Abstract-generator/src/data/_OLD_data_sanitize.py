import __init__  # For now, needed for all the relative imports

import csv
import os

from src.settings import DIR_DATA, DEFAULT_ARXIV_DATASET, VALID_DATASETS


def read_arxiv_data_raw():
    """6
    Returns:
         - fields: list of headings for each data column
         - rows:   list of lists, each containing the data for given row
    """
    datapath = DIR_DATA + os.sep + DEFAULT_ARXIV_DATASET
    with open(datapath, mode='r') as csv_file:
        csvdata = list(csv.reader(csv_file))
        fields = csvdata[0]
        datalines = csvdata[1:]
    return fields, datalines


def clean_list_of_abstracts(to_remove=('\n')):
    """
    TODO cleaning, dealing with strange characters and punctuation
    Uses read_arxiv_data_raw() to return a list of abstracts from the dataset
    Cleaning options:
        - remove newlines (maybe we should keep them so the LM learns the line spacing too...)
    Args:
        to_remove: tuple of strings to be replaced by a space ' '
    """
    fields, datalines = read_arxiv_data_raw()
    assert fields[-1] == 'summary'
    ndata = len(datalines)
    list_of_abstracts = [0] * ndata
    for idx in range(ndata):
        abstract = datalines[idx][-1]
        for removal in to_remove:
            abstract = abstract.replace(removal, ' ')
        list_of_abstracts[idx] = abstract

    return list_of_abstracts


def load_dataset(dataset):
    """
    Core function for mapping a dataset label to the raw files passed to the tokenizer train method
    """
    assert dataset in VALID_DATASETS
    if dataset == 'arxiv':
        datafiles = clean_list_of_abstracts()
        # datafiles = [a.split(' ') for a in datafiles]
        #print(datafiles[0])
    else:
        assert dataset in ['wikitext-2', 'wikitext-103']
        # see https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
        datafiles = [DIR_DATA + os.sep + dataset + os.sep + 'wiki.%s.raw' % a
                     for a in ["test", "train", "valid"]]
    return datafiles


if __name__ == '__main__':
    list_of_abstracts = clean_list_of_abstracts()
    for abstract_idx in range(len(list_of_abstracts)):
        print('Example of abstract entry %d:' % abstract_idx)
        print(list_of_abstracts[abstract_idx])
        print(list_of_abstracts[abstract_idx].split(' '))
