import __init__  # For now, needed for all the relative imports

import math
import os
import time

import torch.nn as nn
import torch
from torchtext.datasets import WikiText2

# need all class definitions for un-pickle
from src.model.transformer_torch import TransformerModel, PositionalEncoding, load_model
from src.model.train_evaluate import train, evaluate
from src.model.model_utils import data_process, batchify, gen_tokenizer_and_vocab
from src.settings import DIR_MODELS

"""
Note: the ipynb has modified version of code below; this should be functionalized and integrated
"""


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, vocab = gen_tokenizer_and_vocab()

    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    ntokens = len(vocab.stoi)  # the size of vocabulary
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)  # or None

    best_val_loss = float("inf")
    epochs = 3  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, device, train_data, ntokens, optimizer, scheduler, criterion, epoch)
        val_loss = evaluate(model, val_data, device, ntokens, criterion)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    # report best model on test set
    test_loss = evaluate(best_model, test_data, device, ntokens, criterion)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    # save best model (two methods)
    # approach 1: save model (class) entirely (uses pickle)
    save_model(model, DIR_MODELS + os.sep + 'model_epoch%d.pth' % epochs, as_pickle=True)
    # approach 2: save model weights
    save_model(model, DIR_MODELS + os.sep + 'model_weights_epoch%d.pth' % epochs, as_pickle=False)
