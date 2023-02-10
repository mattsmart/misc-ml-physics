import __init__

import csv
import feedparser
import numpy as np
import urllib.request
from pathlib import Path

# bunch of tokenizer options
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, Metaspace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer

from src.settings import VALID_TOKENIZATIONS


def arxiv_api(file_path, max_results=11, search_query='all:electron'):

    start = 0
    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start, max_results)

    # Opensearch metadata such as totalResults, startIndex,
    # and itemsPerPage live in the opensearch namespase.
    # Some entry metadata lives in the arXiv namespace.
    # This is a hack to expose both of these namespaces in
    # feedparser v4.1
    # Python 3.8.7 Windows: need to comment out the following two lines

    # feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    # feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    # perform a GET request using the base_url and query
    response = urllib.request.urlopen(base_url + query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    # # print out feed information
    # print('Feed title: %s' % feed.feed.title)
    # print('Feed last updated: %s' % feed.feed.updated)

    # # print opensearch metadata
    # print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
    # print('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
    # print('startIndex for this query: %s'   % feed.feed.opensearch_startindex)

    abstract_list = []

    # Run through each entry, and print out information
    for entry in feed.entries:
        #  print(entry.keys())
        pc = entry.arxiv_primary_category['term']
        tags = [entry.tags[i].term for i in range(len(entry.tags))]
        data_row = [
            entry.id,
            entry.published_parsed,
            entry.published,
            entry.title,
            pc,
            tags,
            entry.summary]
        abstract_list.append(data_row)

    fields = ['id', 'published_parsed', 'published', 'title', 'arxiv_primary_category',
              'tags', 'summary']

    with open(file_path, mode='w', encoding="utf-8") as csv_file:
        write = csv.writer(csv_file, lineterminator='\n')
        write.writerow(fields)
        write.writerows(abstract_list)

    return file_path


def train_custom_tokenizer(dataset, token_model, tknzr_file,
                           vocab_size,
                           vocab=None,
                           pretrain_fast=False,
                           max_input_chars_per_word=None,
                           eos_token=None, bos_token=None,
                           pad_token=None, mask_token=None,
                           unk_token=None):
    """
    Building a Tokenizer using HuggingFace library. The pipeline seems to be:

        - Model           : algorithm that tokenizes, it is a mandatory
                            component. There are only 4 models implemented
                            (BPE, Unigram, WordLevel, WordPiece)
        - Normalizer      : some preprocessing that could happen before, but
                            doesn't necessarily have to
        - Pre-Tokenizer   : splitting the input according to some rules
        - Post-Processing : needing to add some tokens/input after (mostly seems
                            to be eos, bos tokens)
        - Decoder         : certain previous pipeline steps need to be reversed
                            for proper decoding
        - Trainer         : The corresponding training algorithm for the model

    Note : Some pre-processing might need to happen beforehand in previous
            functions (might be easier using pandas before)

    Input
        token_model (str)        : algorithm to use for tokenization
        dataset (class)          : a python iterator that goes through the data
                                    to be used for training
        token_dir (str)          : directory with tokenizers
        vocab_size (int)         : size of the vocabulary to use
        tokenFilename (str)     : filename of particular token we want to
                                    train. Will overwrite previously save files.
        vocab (list of str)      : models other than BPE can use non-mandatory
                                    vocab as input
        max_input_chars_per_word : used for WordPiece

    Output
        tokenizer                : huggingFace Tokenizer object, our fully
                                    trainer tokenizer

    """
    special_token_lst = [pad_token, bos_token, eos_token, mask_token, unk_token]

    # NFKC
    normalizer_lst = []
    pre_tokenizer_lst = [Whitespace, ByteLevel]
    decoder_lst = []

    bos_idx = special_token_lst.index(bos_token)
    eos_idx = special_token_lst.index(eos_token)

    if token_model == 'BPE':
        model = BPE(unk_token=unk_token)
        Trainer = BpeTrainer
    elif token_model == 'Unigram':
        model = Unigram(vocab=vocab)
        Trainer = UnigramTrainer
    elif token_model == 'WordLevel':
        model = WordLevel(unk_token=unk_token, vocab=vocab)
        Trainer = WordLevelTrainer
    elif token_model == 'WordPiece':
        model = WordPiece(unk_token=unk_token, vocab=vocab,
                          max_input_chars_per_word=max_input_chars_per_word)
        Trainer = WordPieceTrainer
    else:
        error_msg = f'Error: token_model ({token_model}) not an algorithm in%s' \
                    % VALID_TOKENIZATIONS
        raise SystemExit(error_msg)

    # instantiation
    tokenizer = Tokenizer(model)

    # Select a tokenization trainer
    if vocab_size is None:
        trainer = Trainer(show_progress=True, special_tokens=special_token_lst)
    else:
        trainer = Trainer(vocab_size=vocab_size,
                          show_progress=True,
                          special_tokens=special_token_lst)

    # Set the normalizer
    tokenizer.normalizer = normalizers.Sequence([fcn() for fcn in normalizer_lst])

    # Set the pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([fcn() for fcn in pre_tokenizer_lst])

    # Set the post-processing
    tokenizer.post_processor = processors.TemplateProcessing(
        single=bos_token + " $A " + eos_token,
        special_tokens=[(bos_token, bos_idx),
                        (eos_token, eos_idx)],
        #  pair=bos_token+" $A "+eos_token" $B:1 "+eos_token+":1",
    )

    # Set the decoder
    if ByteLevel in pre_tokenizer_lst:
        tokenizer.decoder = decoders.ByteLevel()
    if Metaspace in pre_tokenizer_lst:
        tokenizer.decoder = decoders.Metaspace()
    if token_model == 'WordPiece':
        tokenizer.decoder = decoders.WordPiece()

    # creating iterator
    def batch_iterator():
        for i in np.arange(0, len(dataset)):
            yield dataset[i]

    # train call
    tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator(), length=len(dataset))

    if Path(tknzr_file).exists():
        print(f"Warning : overwriting previously save tokenizer with\
                        same filename ( {tknzr_file} ).")
    tokenizer.save(tknzr_file)

    if pretrain_fast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tknzr_file)
    else:
        tokenizer = PreTrainedTokenizer(tokenizer_file=tknzr_file)
    tokenizer.pad_token = pad_token
    tokenizer.mask_token = mask_token

    return tokenizer


def load_tokenizer(tknzr_file, flag_tknzr_fast, pad_token=None, mask_token=None):
    """
    Interestingly, HuggingFace does not allow the base tokenizer to be called.
    This is a bizarre choice, but accordingly we have to look for something else
    , which is why I use the PreTrainedTokenizerFast to wrap the base tokenizer.
    Written in Rust, it's faster than the base tokenizer class, but also lets
    you call the tokenizer as tknzr('text to be tokenized').

    Input
        tknzr_file (str) : .json file of the tokenizer trained previously
        *_tokens (str)  : tokens that are to be used in the corresponding context
                            Some of them are not implemented yet...
    Output
        tknzr     : tokenizer as PreTrainedTokenizerFast class to be passed on
    """
    if flag_tknzr_fast:
        tknzr = PreTrainedTokenizerFast(tokenizer_file=tknzr_file)
    else:
        tknzr = PreTrainedTokenizer(tokenizer_file=tknzr_file)
    tknzr.pad_token = pad_token
    tknzr.mask_token = mask_token

    return tknzr
