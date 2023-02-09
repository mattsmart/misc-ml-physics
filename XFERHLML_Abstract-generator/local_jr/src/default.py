import os

# NOTE : defaults must be in src for the correct structure to work.
ROOT_DIR = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
DATA_DIR = ROOT_DIR + os.sep + 'data'
RAW_DATA_DIR = DATA_DIR + os.sep + 'raw'
PROC_DATA_DIR = DATA_DIR + os.sep + 'processed'
TOK_DIR = DATA_DIR + os.sep + 'tokenizers'
MODEL_DIR = DATA_DIR + os.sep + 'models'

SPECIAL_TOKEN_LIST = {'bos_token' : "<s>"
                    , 'eos_token' : "<\s>"
                    , 'unk_token' : "<unk>"
                    , 'pad_token' : "<pad>"
                    , 'mask_token' : "<mask>"}
