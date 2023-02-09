# from spacy.lang.en import English
import spacy
import re
import pandas as pd
from spacy.tokenizer import _get_regex_pattern
from spacy.matcher import Matcher

df = pd.read_csv("datasets/raw_arxiv_10.csv")
summaries = df.summary.str.replace(r'\n', ' ')
summaries = summaries.str.replace('  ', '')

## For now just replace all LaTeX expressions with placeholder
summaries = summaries.replace("\$[^\$]+\$", "MATHLATEX", regex=True)

## special cases of :
## 1: T < 1e7 K, 3: LaTeX, 4: citations

nlp = spacy.load("en_core_web_sm")

## this handles most, seems to fail only when there is a space in the LaTeX,
# and doesn't separate from following grammatical tokens (e.g. '.',',', ')'), or preceding hypenated words
# re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
# re_token_match = f"({re_token_match}|\$[^\$]+\$)"
# nlp.tokenizer.token_match = re.compile(re_token_match).match

doc = nlp(summaries[3])


print(summaries[3])

for token in doc:
	print(token.text, token.lex_id) # token.vector gives vectors of length 96.

	# [token.lex_id for token in doc[i]]