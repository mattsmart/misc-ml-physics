import os
import sys


DIR_CURRENT = os.path.dirname(__file__)
DIR_ROOT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_ROOT)

DIR_DATA = DIR_ROOT + os.sep + 'saved_data'
DIR_TOKENIZERS = DIR_ROOT + os.sep + 'saved_tokenizers'
DIR_MODELS = DIR_ROOT + os.sep + 'saved_models'

DEFAULT_ARXIV_DATASET = 'raw_arxiv_10.csv'
VALID_DATASETS = ['wikitext-2', 'wikitext-103', 'arxiv']
VALID_TOKENIZATIONS = ['BPE', 'Unigram', 'WordLevel', 'WordPiece']

# determines sequence length for get_batch(); used elsewhere e.g. train()
BPTT = 35  # constant used by OLD_model.py and training notebook (acts as a max token context length)

for core_dir in [DIR_DATA, DIR_TOKENIZERS, DIR_MODELS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)

SPECIAL_TOKEN_LIST = {
    'bos_token': "<s>",
    'eos_token': "<\s>",
    'unk_token': "<unk>",
    'pad_token': "<pad>",
    'mask_token': "<mask>"}
