import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from misc_functions import attention, subsequent_mask, clones

class GPT(nn.Module):

	def __init__(self, decoder, embed, generator):
		super(GPT, self).__init__()
		self.decoder = decoder
		self.embed = embed 
		self.generator = generator

	def forward(self, x, subsq_mask):
		return self.decoder(self.embed(x), subsq_mask)

class Generator(nn.Module):
	"""Define standard linear + softmax generation step."""
	## The last two steps of the Decoder. This generates the probabilities of each word
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab) ## transforms output from d_model size to vocab size

	def forward(self, x):
		## applies projection from d_model space to vocab space
		## applies softmax followed by logarithm along vocab direction (to generate probabilities of each word)
		return F.log_softmax(self.proj(x), dim=-1)



class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, subsq_mask):
		for layer in self.layers: 
			x = layer(x, subsq_mask)

		return self.norm(x)


class LayerNorm(nn.Module):
	"""Construct a layernorm module (See citation for details)."""
	## https://arxiv.org/abs/1607.06450
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		## features = d_model = 512
		## Parameters are tensors which get added to the parameter list
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

 
class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"""Apply residual connection to any sublayer with the same size."""
		## there's some debate about the order of this
		## I think that it doesn't matter that we keep adding, because the final LayerNorms will rescale the tensors
		return x + self.dropout(sublayer(self.norm(x)))

class DecoderLayer(nn.Module):
	"""docstring for DecoderLayer"""
	def __init__(self, size, attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.attn = attn
		self.feed_forward = feed_forward
		self.dropout = dropout
		self.sublayer = clones(SublayerConnection(size, dropout), 2)

	def forward(self, x, subsq_mask):

		x = self.sublayer[0](x, lambda x: self.attn(x, x, x, subsq_mask))

		return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"""Take in model size and number of heads."""
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4) # input size = 512, output size = 512
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		"""Implements Figure 2"""
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)  
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 

		## applies linear transformation to query, key, values
		## then reshapes to [30, -1, 8, 512//8], and transposes to [30, 8, -1, 512//8] (-1 is 9 or 10)
		## the reshape, then transpose conserves the ordering of "words" within the "sentence"
		## attn is calculated independently for each Nth 64 elements of the embeddings of a sentence
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
			 for l, x in zip(self.linears, (query, key, value))] # 1st arg to zip are 4 nn.Linears, this only uses the first 3

		## masks are only applied at this step, but haven't positions already been 'scrambled'
		## by previous linear transformation?
		x, self.attn = attention(query, key, value, mask=mask, 
								 dropout=self.dropout)
		
		# 3) "Concat" using a view and apply a final linear. 
		## contiguous() copies memory
		## stiches weighted values back from [30, 8, 10, 64] to [30, 10, 512]
		x = x.transpose(1, 2).contiguous() \
			 .view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x) # the last nn.Linears is used here, I guess.
		

class PositionwiseFeedForward(nn.Module):
	"""Implements FFN equation."""
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		## vocab is vocabulary size (=11)
		## d_model is dimensionality of each embedding vector (=512)
		self.lut = nn.Embedding(vocab, d_model) 
		self.d_model = d_model

	def forward(self, x): # x needs to be a LongTensor
		## x.shape = [30, 10] or [30, 9]
		## self.lut(x).shape = [30, 10, 512] or [30, 9, 512]
		return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
	"""Implement the PE function."""
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model, requires_grad=False) # size: [5000, 512]
		position = torch.arange(0, max_len).unsqueeze(1) # size: [5000, 1]
		## not sure why this is implemented "in log space"
		div_term = torch.exp(torch.arange(0, d_model, 2) *
							 -(math.log(10000.0) / d_model)) # exp{[0, 2, 4, ..., 510]*-ln(10000)/512}
		pe[:, 0::2] = torch.sin(position * div_term) # every second column, starting from 0
		pe[:, 1::2] = torch.cos(position * div_term) # every second column, starting from 1
		pe = pe.unsqueeze(0) # shape: [1, 5000, 512]

		## registers pe as a buffer that should not to be considered a model parameter.
		## Buffers, by default, are persistent and will be saved alongside parameters.
		## Often used for running averages
		self.register_buffer('pe', pe)
		
	def forward(self, x): 
		## takes normalized, embedded x

		# x.shape is [30, 10, 512] or [30, 9, 512]
		# pe added is [1, 10, 512] or [1, 9, 512]
		x = x + self.pe[:, :x.size(1)] 
		return self.dropout(x) # applies dropout (zeros some elements of x with prob=dropout)