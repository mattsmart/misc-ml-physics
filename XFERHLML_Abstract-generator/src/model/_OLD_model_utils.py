import __init__  # For now, needed for all the relative imports

import torch
from collections import Counter
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.legacy.vocab import Vocab as RetiredVocab

from src.settings import BPTT


def gen_tokenizer_and_vocab():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = RetiredVocab(counter)
    return tokenizer, vocab


def data_process(raw_text_iter, vocab, tokenizer):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz, device):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(BPTT, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target
