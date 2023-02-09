import __init__  # For now, needed for all the relative imports

import math
import time

import numpy as np
import torch

from src.model._OLD_model_utils import get_batch  # TODO remove need for this import in train()
from src.settings import BPTT


def train(model, device, train_data, ntokens, optimizer, scheduler, criterion, epoch):
    """
    scheduler: either an int/float (fixed learning rate) or a scheduler torch object
    """
    model.train()  # set to: evaluation mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(BPTT).to(device)

    batch_indices = np.arange(0, train_data.size(0) - 1, BPTT)
    loss_per_batch = 0.0 * batch_indices  # record the training loss for each batch

    for batch, i in enumerate(batch_indices):
        src, tgt = get_batch(train_data, i)
        optimizer.zero_grad()
        if src.size(0) != BPTT:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
        output = model(src, src_mask)
        loss = criterion(output.view(-1, ntokens), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # TODO check if scaling is correct
        loss_per_batch[batch] = loss

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if isinstance(scheduler, int) or isinstance(scheduler, float):
                last_lr = scheduler
            else:
                last_lr = scheduler.get_last_lr()[0]

            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:2e} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch,
                                                      batch, len(train_data) // BPTT,
                                                      last_lr,
                                                      elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return loss_per_batch


def train_version_jeremy(model, dataloader, device, vocab_size, epoch, optimizer, scheduler, criterion, max_len=None):
    """
    Training loop that takes batches from dataloader and pushes them to device to train.
    Will check if they're the same size of max_len: if shorter, will reduce to the longest length in the batch.
    Then trains according to optimizer, criterion, and schedule.

    Input
        model (instance)        : model that is being trained
        dataloader (instance)   : dataloader that batches data into tensors
        optimizer (instance)    : Not sure what type optimizers are
        criterion               :
        device (str)            : gpu or cpu
        max_len (int)           : maximum sentence length if not None
    Output
        None
    """
    model.train()  # set to: evaluation mode
    total_loss = 0.
    start_time = time.time()
    if max_len is not None:
        src_mask = model.generate_square_subsequent_mask(max_len).to(device)

    batch_indices = np.arange(0, len(dataloader))
    loss_per_batch = 0.0 * batch_indices  # record the training loss for each batch

    for i, batch in enumerate(dataloader):
        # print((batch.src).is_pinned())
        src = (batch.src).to(device)
        tgt = (batch.tgt).to(device)
        src_pad_mask = (batch.src_pad_mask).to(device)
        # tgt_pad_mask = (batch.tgt_pad_mask).to(device)

        optimizer.zero_grad()
        if src.size(0) != max_len:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)

        output = model(src, src_mask, src_key_padding_mask=src_pad_mask)
        loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))
        loss.backward()
        torch.torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_per_batch[i] = loss  # TODO check if scaling is correct
        total_loss += loss.item()
        log_interval = 200
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, i, len(dataloader),
                                                      scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / log_interval,
                                                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return loss_per_batch


def evaluate(eval_model, data_source, device, ntokens, criterion):
    eval_model.eval()  # set to: evaluation mode
    total_loss = 0.
    src_mask = eval_model.generate_square_subsequent_mask(BPTT).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, BPTT):
            src, tgt = get_batch(data_source, i)
            if src.size(0) != BPTT:
                src_mask = eval_model.generate_square_subsequent_mask(src.size(0)).to(device)
            output = eval_model(src, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(src) * criterion(output_flat, tgt).item()
    return total_loss / (len(data_source) - 1)


def evaluate_version_jeremy(eval_model, dataloader, device, vocab_size, criterion, max_len):
    """
    Takes a trained model, puts it in evaluation mode to see how well it
    performs on another set of data.

    Input
        eval_model (instance)   : model to be evaluated
        max_len (int)           : maximum length possible/trained on
        dataloader (instance)   : dataloader of the dataset that is used for evaluation
        nbr_samples (int)       : Supposed to be number of samples [Jeremy note: not sure I need]
    Output
        loss of evaluated set
    """
    eval_model.eval()  # set to: evaluation mode
    total_loss = 0.
    total_len = 0.
    src_mask = eval_model.generate_square_subsequent_mask(max_len).to(device)
    with torch.no_grad():
        for batch in dataloader:
            src = (batch.src).to(device)  # dim(src) = sentence_len x nbr_samples
            tgt = (batch.tgt).to(device)
            src_pad_mask = (batch.src_pad_mask).to(device)
            if src.size(0) != max_len:
                src_mask = eval_model.generate_square_subsequent_mask(src.size(0)).to(device)
            output = eval_model(src, src_mask, src_key_padding_mask=src_pad_mask)
            output_flat = output.view(-1, vocab_size)
            total_loss += src.size(0) * criterion(output_flat, tgt.reshape(-1)).item()
            total_len += src.size(0)
    return total_loss / (total_len - 1)  # TODO : why -1 ?
