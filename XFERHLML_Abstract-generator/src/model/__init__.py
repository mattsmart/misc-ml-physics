import os
import sys

DIR_CURRENT = os.path.dirname(__file__)
DIR_SRC = os.path.dirname(DIR_CURRENT)
DIR_ROOT = os.path.dirname(DIR_SRC)

sys.path.append(DIR_ROOT)
