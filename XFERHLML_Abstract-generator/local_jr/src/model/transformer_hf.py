import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def pad_tensor(vec, padSize, dim, pad):
    """
    Input:
        vec : tensor to pad
        padSize : the size to pad to
        dim : dimension to pad
        pad : value of pad

    Output:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = padSize - vec.size(dim)
    return torch.cat([vec, pad * torch.ones(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or maxLen
    """

    def __init__(self, dim=0, maxLen=100, padValue=0):
        """
        Input:
            dim : the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.maxLen = maxLen
        self.padValue = padValue

    def pad_collate(self, batch):
        """
        Input:
            batch : list of (tensor, label)

        Output:
            xs : a tensor of all examples in 'batch' after padding
            ys : a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len_seq = np.max( [ x.shape[self.dim] for x in batch ] )
        max_len = np.min( [max_len_seq, self.maxLen] )

        # pad according to max_len
        batch = [pad_tensor(x[:max_len], padSize=max_len, pad=self.padValue
                                , dim=self.dim) for x in batch ]
        # stack all
        data = torch.stack([x[:-1] for x in batch], dim=1) # change to dim = 0 for annotated transformer?
        target = torch.stack([x[1:] for x in batch], dim=1)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))
        return [data.long(), target.long()]

    def __call__(self, batch):
        return self.pad_collate(batch)

class PadCollateMemoryOnGPU:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or maxLen
    """

    def __init__(self, dim=0, maxLen=100, padValue=0):
        """
        Input:
            dim : the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.maxLen = maxLen
        self.padValue = padValue

    def pad_collate(self, batch):
        """
        Input:
            batch : list of (tensor, label)

        Output:
            xs : a tensor of all examples in 'batch' after padding
            ys : a LongTensor of all labels in batch
        """
        # find longest sequence
        print(batch[0])
        max_len_seq = np.max( [ x.shape[self.dim] for x in batch ] )
        max_len = np.min( [max_len_seq, self.maxLen] )

        # pad according to max_len
        batch = [pad_tensor(x[:max_len], padSize=max_len, pad=self.padValue
                                , dim=self.dim) for x in batch ]
        # stack all
        data = torch.stack([x[:-1] for x in batch], dim=1) # change to dim = 0 for annotated transformer?
        target = torch.stack([x[1:] for x in batch], dim=1)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))
        return [data.long(), target.long()]

    def __call__(self, batch):
        return self.pad_collate(batch)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask=None): # should I add a padding mask here?
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
