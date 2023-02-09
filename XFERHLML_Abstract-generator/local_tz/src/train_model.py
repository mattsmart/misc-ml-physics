import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc_functions import attention, subsequent_mask
from gpt_model import *
import math, copy, time


def make_model(vocab, N=12, 
			   d_model=512, d_ff=2048, h=8, dropout=0.1):
	"""Helper: Construct a model from hyperparameters."""

	## returns EncoderDecoder object
	c = copy.deepcopy
	attn = MultiHeadedAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = GPT(Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N),
		## Sequential passes input to the forward() method in the first module it stores
		## and then "chains" outputs to inputs sequentially for subsequent modules,
		nn.Sequential(Embeddings(d_model, vocab), c(position)),
		Generator(d_model, vocab))
	
	# This was important from their code. 
	# Initialize parameters with Glorot / fan_avg.
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p) # what does this do? How does it modify model?
	return model


# class Batch:
# 	"""Object for holding a batch of data with mask during training."""

# 	## takes src, makes src mask (for padding)

# 	## takes trg, make right and left shifts
# 	## make trg mask to hide padding and future words

class Batch:
	"""Object for holding a batch of data with mask during training."""
	def __init__(self, trg, pad=0):

		self.trg = trg[:, :-1] # cuts out last column of trg (why?)
		self.trg_y = trg[:, 1:] # cuts out first colum of trg (all 1s)
		self.trg_mask = \
			self.make_std_mask(self.trg, pad)

		# print(self.trg_mask[0])
		self.ntokens = (self.trg_y != pad).data.sum() # 270 = 30 * 9
		# print(self.ntokens)


	@staticmethod
	def make_std_mask(tgt, pad):
		"""Create a mask to hide padding and future words."""
		## tgt has shape [30, 9], after unsqueeze has shape [30, 1, 9]
		tgt_mask = (tgt != pad).unsqueeze(-2)
		# set type of return value of subsequent_mask to same as tgt_mask.data
		
		## subsequent_mask is bool tensor with upper diagonal set to False (shape [1, 9, 9])
		## tgt_mask is true wherever tgt is not equal to pad
		tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
		# & takes intersection of two sets, final shape is [30, 9, 9]
		return tgt_mask

# def run_epoch():

# 	## logging 

# 	## iterates over batches
# 	out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

# 	## compute loss on left shifted trg
# 	## loss computation includes optimizer and training step

def run_epoch(data_iter, model, loss_compute):
	"""Standard Training and Logging Function"""
	start = time.time()
	total_tokens = 0
	total_loss = 0
	tokens = 0
	for i, batch in enumerate(data_iter):
		out = model.forward(batch.trg, batch.trg_mask)
		print(batch.trg[0])
		print(batch.trg_mask[0])
		loss = loss_compute(out, batch.trg_y, batch.ntokens)
		total_loss += loss
		total_tokens += batch.ntokens
		tokens += batch.ntokens
		if i % 50 == 1:
			elapsed = time.time() - start
			print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
					(i, loss / batch.ntokens, tokens / elapsed))
			start = time.time()
			tokens = 0
	return total_loss / total_tokens


class NoamOpt:
	#"Optim wrapper that implements rate."
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0
		
	def step(self):
		# "Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optimizer.step()
		
	def rate(self, step = None):
		# "Implement `lrate` above"
		if step is None:
			step = self._step
		return self.factor * \
			(self.model_size ** (-0.5) *
			min(step ** (-0.5), step * self.warmup ** (-1.5)))
		
def get_std_opt(model):
	return NoamOpt(model.embed[0].d_model, 2, 4000,
			torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
	# "Implement label smoothing."
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False) # Kullback-Leibler divergence loss
		self.padding_idx = padding_idx
		self.confidence = 1.0 - smoothing
		self.smoothing = smoothing
		self.size = size
		self.true_dist = None
		
	def forward(self, x, target):
		assert x.size(1) == self.size
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0
		mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist.requires_grad_(False)
		# return self.criterion(x, Variable(true_dist, requires_grad=False))
		return self.criterion(x, true_dist)

def data_gen(V, batch, nbatches):
	# "Generate random data for a src-tgt copy task."
	# generates nbatches of Batch objects

	for i in range(nbatches):

		A = np.tile(np.arange(1,11), batch)
		A = A.reshape(batch, 10)

		rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

		## Use always a negative shift, so that column_indices are valid.
		## (could also use module operation)
		r = np.random.randint(-9, 1, size=batch)
		r[r < 0] += A.shape[1]
		column_indices = column_indices - r[:, np.newaxis]

		result = A[rows, column_indices]

		## Add random padding
		num_pad = np.random.poisson(2, size=batch)
		# print(num_pad)
		for j in range(batch):
			if num_pad[j] == 0:
				pass
			else:
				result[j, -num_pad[j]:] = 0

		data = torch.from_numpy(result)
		tgt = data
		# print(tgt)
		yield Batch(tgt, 0)

class SimpleLossCompute:
	# "A simple loss compute and train function."
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator
		self.criterion = criterion # LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
		self.opt = opt # NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
		
	def __call__(self, x, y, norm):
		x = self.generator(x) # x is output, each element now in d_vocab dimensions, shape = [30, 9, 11]
							  # y is batch.trg_y (first column of 1s removed), shape = [30, 9]
							  # norm is batch.ntokens (270)

		loss = self.criterion(x.contiguous().view(-1, x.size(-1)), # shape = [270, 11]
							  y.contiguous().view(-1)) / norm # shape = [270]
		loss.backward() # compute gradients (of what?)
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad()

		if list(loss.data.size()) != []:
			return loss.data[0] * norm
		else:
			return loss.data * norm


V = 11 # input symbols are integers from 1 to 11 inclusive
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, N=2)
## uses pytorch's Adam optimizer
model_opt = NoamOpt(model.embed[0].d_model, 1, 400,
		torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
	model.train() ## calls nn.Module.train() which sets mode to train
	run_epoch(data_gen(V, 30, 20), model, # generates 20 batches of [30, 10] random integers (first column is 1)
			  SimpleLossCompute(model.generator, criterion, model_opt))
	model.eval() ## sets mode to testing (i.e. train=False). Layers like dropout behave differently depending on if mode is train or testing.
	run_epoch(data_gen(V, 30, 5), model, 
					SimpleLossCompute(model.generator, criterion, None))

def greedy_decode(model, max_len, start_symbol):
	ys = torch.ones(1, 1).fill_(start_symbol).long()
	for i in range(max_len-1):
		# print(ys)
		out = model.forward(ys, subsequent_mask(ys.size(1)))
		prob = model.generator(out[:, -1])
		_, next_word = torch.max(prob, dim = 1)
		next_word = next_word.data[0]
		ys = torch.cat([ys, 
						torch.ones(1, 1).long().fill_(next_word)], dim=1)
	return ys


## Let's see what trained output looks like 
model.eval()
verbose = True
print(greedy_decode(model, max_len=10, start_symbol=0))
print(greedy_decode(model, max_len=10, start_symbol=1))
print(greedy_decode(model, max_len=10, start_symbol=4))
print(greedy_decode(model, max_len=10, start_symbol=9))
