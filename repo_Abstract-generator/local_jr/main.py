import os, torch, time, math, sys, re, csv
import numpy as np

sys.path.append('..' + os.sep )
from src import default
import src.data.dataset_class as dsc
import src.data.dataloader_class as dlc

from src.model.transformer_hf import TransformerModel
from src.model.generate_text import gen_some_text
from src.model.train_evaluate import train, evaluate
#from src.model.transformer import make_gpt_model # imports don't work

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### DATA
# create dataset
dataset = dsc.ArxivDataset()
#dataset = dsc.WikiTextDataset()

#train tokenizer (or use one already trained)
tknzrType = 'BPE'
tknzrTrain = True
tknzrFast = True

_ = dataset.tokenizer(tknzrTrain, tknzrType, tknzrFast=tknzrFast)

dataloader = dlc.CustomDataloader(dataset, batchSize, maxLen)

# transformer from huggingface

### MODEL
maxLen     = 40 # maximum sentence length
vocabSize  = None # None if you want to let tokenizer do its thing
emsize     = 512 # embedding dimension
nhid       = 2048 # dimension of feedforward net in torch.nn.TransformerEncoder
nlayers    = 12 # number of TransformerEncoderLayer in TransformerEncoder
nhead      = 8 # number of heads in the multiheadattention models
dropout    = 0.2 # the dropout value
batchSize = 10 #32
valBatchSize = 10 #32, not used right now.
epochs     = 50  # The number of epochs

TRAIN = True
# TODO : Change to the Annotated Transformer if I want (dim=1)
model = TransformerModel(dataset.vocabSize, emsize, nhead, nhid, nlayers,
                                dropout).to(device)
# criterion
criterion = torch.nn.CrossEntropyLoss()#ignore_index=tknzr.get_vocab()["<pad>"])

# optimizer
paramsAdam  = [{'params' : model.parameters(), 'lr' : 1e-3,
                'betas' : (0.9, 0.999), 'eps' : 1e-08, 'weight_decay' : 0.0}]
paramsAdamW = [{'params' : model.parameters(), 'lr' : 5e-5,
                'betas' : (0.9, 0.999), 'eps' : 1e-08, 'weight_decay' : 0.0}]
paramsSGD   = [{'params' : model.parameters(), 'lr' : 0.5, 'momentum' : 0.0,
                'dampening' : 0.0, 'weight_decay' : 0.0}]

#optimizer = torch.optim.SGD( paramsSGD )
#optimizer = torch.optim.Adam( paramsAdam )
optimizer = torch.optim.AdamW( paramsAdamW )

# scheduler, 1.0 to signify no decay rate
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

### TRAINING

# fasttokenizer should not be used before forking. Something
# to figure out. What this does is suppress some warning messages
# https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
# doesn't seem to affect the timing though
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if TRAIN:
    best_val_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train( model, dataloader.train, device, dataset.vocabSize, epoch,
                    optimizer, scheduler, criterion, maxLen)
        val_loss = evaluate(model, dataloader.valid, device, dataset.vocabSize,
                                criterion, maxLen, len(dataloader.dsetValid))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time()-epoch_start_time),
                                         val_loss, math.exp(val_loss)))
                                         # Why is math.exp so large????
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    # save best model (two methods)
    modelFull = default.MODEL_DIR + os.sep + f'{dataset.name}_epoch{epochs}.pth'
    modelWeights = default.MODEL_DIR + os.sep\
                        + f'{dataset.name}_weights_epoch{epochs}.pth'
    modelFullBest = default.MODEL_DIR + os.sep\
                        + f'{dataset.name}_epoch{epochs}_best.pth'
    modelWeightsBest = default.MODEL_DIR + os.sep\
                        + f'{dataset.name}_weights_epoch{epochs}_best.pth'
    # approach 1: save model (class) entirely (uses pickle)
    torch.save(model, modelFull)
    torch.save(best_model, modelFullBest)
    # approach 2: save model weights
    torch.save(best_model.state_dict(), modelWeightsBest)
