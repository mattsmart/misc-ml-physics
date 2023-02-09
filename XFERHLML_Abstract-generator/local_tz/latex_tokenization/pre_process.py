# from spacy.lang.en import English
import spacy
import re
import pandas as pd
from spacy.tokenizer import _get_regex_pattern
from spacy.matcher import Matcher

df = pd.read_csv("datasets/raw_arxiv_10.csv")
summaries = df.summary.str.replace(r'\n', ' ')
summaries = summaries.str.replace('  ', '')

## special cases:
## 1: T < 1e7 K, 3: LaTeX, 4: citations

## spaCy splits up hyphenated words "wave-particle" -> "wave" + "-" + "particle". Can remove from infix if needed.


nlp = spacy.load("en_core_web_sm")



# print(summaries[3])


## this handles most, seems to fail only when there is a space in the LaTeX,
# and doesn't separate from following grammatical tokens (e.g. '.',',', ')'), or preceding hypenated words
re_token_match = _get_regex_pattern(nlp.Defaults.token_match)
# re_token_match = f"({re_token_match}|\$.*?\$)"
re_token_match = f"({re_token_match}|\$[^\$]+\$)"
nlp.tokenizer.token_match = re.compile(re_token_match).match

print(nlp.tokenizer.explain('$\beta \gamma$'))

latex = r"\$[^\$]+\$"

doc = nlp(summaries[3])


# for match in re.finditer(latex, doc.text):
# 	start, end = match.span()
# 	span = doc.char_span(start, end)
# 	if span is not None:
# 		print("Found match:", span.text)

print(summaries[3])

for token in doc:
	print(token.text) # token.vector gives vectors of length 96.