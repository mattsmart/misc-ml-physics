import __init__  # For now, needed for all the relative imports

import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

from src.data._OLD_data_sanitize import load_dataset
from src.settings import DIR_DATA, DIR_TOKENIZERS, VALID_DATASETS, VALID_TOKENIZATIONS

"""
Note: huggingface has two relevant classes: Tokenizer and Transformer
- Tokenizer referred to here as 'raw tokenizer'
- Transformer can act as a wrapper for Tokenizer functionality ('not raw tokenizer')

Option 1: use pre-trained tokenizer
Option 2: train own tokenizer

Tokenizer flavors: https://huggingface.co/docs/tokenizers/python/latest/components.html#models
See also:
    https://huggingface.co/docs/tokenizers/python/latest/
    https://huggingface.co/transformers/tokenizer_summary.html
- WordLevel
    - This is the “classic” tokenization algorithm. Map words to IDs without anything fancy.
    Using this Model requires the use of a PreTokenizer.
    No choice will be made by this model directly, it simply maps input tokens to IDs
- BPE
    - One of the most popular subword tokenization algorithm.
    The Byte-Pair-Encoding works by starting with characters, while merging those that are
    the most frequently seen together, thus creating new tokens.
    It then works iteratively to build new tokens out of the most frequent pairs in a corpus.
    - BPE is able to build words it has never seen by using multiple subword tokens,
    and thus requires smaller vocabularies, with less chances of having “unk” (unknown) tokens.
- WordPiece
    - This is a subword tokenization algorithm quite similar to BPE, used mainly by Google in
    models like BERT. It uses a greedy algorithm, that tries to build long words first,
    splitting in multiple tokens when entire words don’t exist in the vocabulary.
    This is different from BPE that starts from characters, building bigger tokens as possible.
    - Uses the famous ## prefix to ID tokens that are part of a word (ie not starting a word).
- Unigram
    - Unigram is also a subword tokenization algorithm, and works by trying to identify the best
    set of subword tokens to maximize the probability for a given sentence. This is different
    from BPE in the way that this is not deterministic based on a set of rules applied
    sequentially. Instead Unigram will be able to compute multiple ways of tokenizing,
    while choosing the most probable one.
"""


def default_tpath(dataset, style):
    tpath = DIR_TOKENIZERS + os.sep + '%s_%s.json' % (style, dataset)
    return tpath


def train_tokenizer_vocab(dataset, style='BPE', force_retrain=True):
    """
    if force_retrain: overwrite the stored tokenizer from tokenizers dir (by retraining)
    else: load the tokenizer if it exists
    """
    assert dataset in VALID_DATASETS
    assert style in VALID_TOKENIZATIONS

    tpath_expected = default_tpath(dataset, style)

    train = True
    if not force_retrain and os.path.isfile(tpath_expected):
        tokenizer = Tokenizer.from_file(tpath_expected)
        train = False
    else:
        print('%s tokenizer file does not exist; training new tokenizer' % tpath_expected)

    if train:

        # load data associated with one of the valid datasets (from /data/ directory)
        datafiles = load_dataset(dataset)

        # Steps for each algo (e.g. BPE):
        # - init Tokenizer using algo
        # - specify algo specific trainer
        # - specify any pre-processing of text (will affect decoding)
        #   see: https://huggingface.co/docs/tokenizers/python/latest/components.html#decoders
        # - different training calls if its the arxiv dataset or wikitext
        #   see https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

        if style == 'BPE':
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.pre_tokenizer = ByteLevel()
            if dataset == 'arxiv':
                tokenizer.train_from_iterator(datafiles, trainer=trainer)
            else:
                tokenizer.train(datafiles, trainer=trainer)
            tokenizer.decoder = decoders.ByteLevel()

        else:
            assert style == 'WordLevel'
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.pre_tokenizer = Whitespace()
            if dataset == 'arxiv':
                tokenizer.train_from_iterator(datafiles, trainer=trainer)
            else:
                tokenizer.train(datafiles, trainer=trainer)
            tokenizer.decoder = decoders.WordPiece()  # WordPiece seems to work (adds back spaces)

        # Save to tokenizers directory
        tokenizer.save(tpath_expected)

    # Generate vocab object based on tokenizer.decoder() method
    # ... TODO implement the same vocabulary functionality, or ensure it is present in Tokenizer and then code it elsewhere...
    # Features we need to match:
    #   from torchtext.legacy.vocab import Vocab as RetiredVocab
    #   ntokens = len(vocab.stoi) ---> ntokens = tokenizer.(...)
    #   data = [torch.tensor([vocab[token] for token in tokenizer(item)],
    #                         dtype=torch.long) for item in raw_text_iter]
    #   tokenized_text_ints = torch.tensor([vocab[token] for token in tokenized_text], dtype=torch.long)
    #   running_context_string = ' '.join([vocab.itos[src[k]] for k in range(src.shape[0])])
    #   unk_index = vocab.unk_index
    vocab = None

    return tokenizer, vocab


"""
def train_BPE(use_arxiv=False, outpath=None):
    # TODO merge into train_tokenizer general function
    # currently: arxiv or wiki dataset

    # specify algo
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # specify algo specific trianer
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # pre-processing before training; will also affect the decoding step
    # see: https://huggingface.co/docs/tokenizers/python/latest/components.html#decoders
    tokenizer.pre_tokenizer = ByteLevel()

    # specify dataset and train
    if use_arxiv:
        abstracts = clean_list_of_abstracts()
        # abstracts = [a.split(' ') for a in abstracts]  # TODO this may be done by pre_tokenizer
        print(abstracts[0])
        tokenizer.train_from_iterator(abstracts, trainer=trainer)
    else:
        # see https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
        files = [DIR_DATA + os.sep + 'wikitext-103' + os.sep + 'wiki.%s.raw' % a
                 for a in ["test", "train", "valid"]]
        tokenizer.train(files, trainer=trainer)

    tokenizer.decoder = decoders.ByteLevel()

    if outpath is not None:
        tokenizer.save(outpath)

    return tokenizer
"""

def train_wordpiece_bert():
    """
    Sample code from: https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
    """
    from tokenizers.models import WordPiece
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    from tokenizers import normalizers
    from tokenizers.normalizers import Lowercase, NFD, StripAccents
    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    from tokenizers.pre_tokenizers import Whitespace
    bert_tokenizer.pre_tokenizer = Whitespace()

    from tokenizers.processors import TemplateProcessing
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    bert_tokenizer.decoder = decoders.WordPiece()

    from tokenizers.trainers import WordPieceTrainer
    trainer = WordPieceTrainer(
        vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    files = [DIR_DATA + os.sep + 'wikitext-103' + os.sep + 'wiki.%s.raw' % a
             for a in ["test", "train", "valid"]]
    bert_tokenizer.train(files, trainer)
    bert_tokenizer.save(DIR_TOKENIZERS + os.sep + 'bert_wiki.json')

    return bert_tokenizer


def tokenizer_examples(tokenizer, raw_tokenizer=True, title='default'):
    """
    Example of a "Raw tokenizer":
        tokenizer = Tokenizer.from_file(tpath)
    Example of "not Raw tokenizer":
        from transformers import PreTrainedTokenizerFast
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tpath)
    """
    title_break = '\n****************************************************************'
    # example text
    text0 = "Hello, y'all! How are you 😁 ?"
    text1 = "Here is some code spaghetti"
    text2 = "configuration interaction (CI) wave functions is examined"
    text3 = "By analogy with the pseudopotential approach for electron-ion interactions"
    text4 = "Welcome to the 🤗 Tokenizers library."
    examples = [text0, text1, text2, text3, text4]

    if raw_tokenizer:
        print('Tokenizer examples (raw_tokenizer=True): %s%s' % (title, title_break))
        for idx, text in enumerate(examples):
            pre = '(Ex %d)' % idx
            print('%s input: %s' % (pre, text))
            output = tokenizer.encode(text)
            print('%s output type & output.tokens: %s, %s' % (pre, type(output), output.tokens))
            print('%s decode(output.ids): %s' % (pre, tokenizer.decode(output.ids)))
            print()
    else:
        print('Tokenizer examples (raw_tokenizer=False): %s%s' % (title, title_break))
        for idx, text in enumerate(examples):
            pre = '(Ex %d)' % idx
            print('%s input: %s' % (idx, text))
            output = tokenizer.encode(text)
            print('%s output type & output: %s, %s' % (pre, type(output), output))
            print('%s decode w/ no cleanup: %s' %
                  (pre, tokenizer.decode(output, clean_up_tokenization_spaces=False)))
            print('%s decode w/ cleanup: %s' %
                  (pre, tokenizer.decode(output, clean_up_tokenization_spaces=True)))
            print()
    return


if __name__ == '__main__':
    style = 'WordLevel'
    dataset = 'wikitext-2'
    tpath = default_tpath(dataset, style)
    tokenizer, vocab = train_tokenizer_vocab(dataset, style=style, force_retrain=True)
    #tokenizer_examples(tokenizer, raw_tokenizer=True, title='default_raw')

    from transformers import PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tpath)
    tokenizer_examples(fast_tokenizer, raw_tokenizer=False, title='default_notraw')


"""
    flag_retrain = False
    use_arxiv = False

    if use_arxiv:
        tpath = DIR_TOKENIZERS + os.sep + 'BPE_arxiv.json'
    else:
        tpath = DIR_TOKENIZERS + os.sep + 'BPE_wiki.json'

    if flag_retrain:
        tokenizer = train_BPE(use_arxiv=use_arxiv, outpath=tpath)
    else:
        tokenizer = Tokenizer.from_file(tpath)

    # raw tokenizer example
    tokenizer_examples(tokenizer, raw_tokenizer=True, title='trained_BPE')

    # try using their tokenizer wrapper for transformers module with "cleanup" support
    from transformers import PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tpath)
    tokenizer_examples(fast_tokenizer, raw_tokenizer=False, title='trained_BPE_wrapper')

    # try pre-trained bert
    from tokenizers import BertWordPieceTokenizer
    bert_tokenizer = BertWordPieceTokenizer(
        DIR_TOKENIZERS + os.sep + "bert-base-uncased-vocab.txt", lowercase=True)
    tokenizer_examples(bert_tokenizer, raw_tokenizer=True, title='pretrained_BERT_wordpiece')

    # train own bert see: https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
    if flag_retrain:
        bert_tokenizer_trained = train_wordpiece_bert()
    else:
        bert_tokenizer_trained = Tokenizer.from_file(DIR_TOKENIZERS + os.sep + 'bert_wiki.json')
    tokenizer_examples(bert_tokenizer_trained, raw_tokenizer=True, title='BERT_wordpiece')
    """