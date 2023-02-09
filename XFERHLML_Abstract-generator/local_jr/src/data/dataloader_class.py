import math
import numpy as np

import torch
from torch.utils.data import DataLoader

class CustomBatch():
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or maxLen. Unclear to me whether this is instantiated
    at every call to dataloader or if it's instantiated along with dataloader.
    For now, all the tensors exist on CPU and are later pushed to the GPU. Needs
    potentially to be changed.
    """

    def __init__(self, data, dim=0, padValue=0, stackDim=1, maxLenModel=None,):
        """
        Input:
            data (dataset)      : a batch of dataset.
            dim (int)           : the dimension to be padded (dimension of time
                                    in sequences)
            maxLenModel (int)   : maixmum length of sentence, if any
            padValue (int)      : the value for padding.
            stackDim (int)      : dimension along which to stack the data tensor.
                                    1 - huggingface, 0 - annotated transformer
        """
        self.dim = dim; self.padValue = padValue

        max_len_seq = np.max( [ x.shape[self.dim] for x in data ] )
        self.maxLen = np.min( [ x for x in [max_len_seq, maxLenModel]
                                                            if x is not None ] )
        # pad according to max_len
        batch = [self.pad_tensor(x[:self.maxLen]) for x in data ]
        # stack all, change to dim = 0 for annotated transformer?
        self.src = (torch.stack([x[:-1] for x in batch], dim=stackDim)).long()
        self.tgt = (torch.stack([x[1:] for x in batch], dim=stackDim)).long()
        self.src_pad_mask = (self.src == self.padValue)
        #self.tgt_pad_mask = (self.tgt != self.padValue).type(torch.int)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))

    def pad_tensor(self, vec):
        """
        Padding a tensors to the max length in the batch.
        Input:
            vec : tensor to pad
        Output:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        padSize = list(vec.shape)
        padSize[self.dim] = self.maxLen - vec.size(self.dim)
        return torch.cat([vec, self.padValue*torch.ones(*padSize)],dim=self.dim)

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

class CustomDataloader():
    def __init__(self, dataset, batchSize, maxLen, dim=0, numWorkers=2,
                    pinMemory=True, trainTestVal=(0.7,0.2,0.1)):

        trainTestVal = [ np.floor(trainTestVal[0]*len(dataset))\
                    , np.floor(trainTestVal[1]*len(dataset))\
                    , len(dataset) - ( np.floor( trainTestVal[0]*len(dataset) )
                    + np.floor( trainTestVal[1]*len(dataset) ) )
                    ]
        self.dsetTrain, self.dsetTest, self.dsetValid =\
                        torch.utils.data.random_split(dataset,
                                [int(x) for x in trainTestVal],
                                generator=torch.Generator().manual_seed(42)
                                )

        tknzr = dataset.transform

        def collate_wrapper(batch):
            return CustomBatch(batch, dim=dim, maxLenModel=maxLen,
                                padValue=tknzr.get_vocab()["<pad>"])

        self.train = DataLoader(self.dsetTrain, batch_size=batchSize,
                                        shuffle=True, num_workers=numWorkers,
                                        collate_fn=collate_wrapper,
                                        pin_memory=pinMemory
                                        )
        self.test = DataLoader(self.dsetTest, batch_size=batchSize,
                                        shuffle=True, num_workers=numWorkers,
                                        collate_fn=collate_wrapper,
                                        pin_memory=pinMemory
                                        )
        self.valid = DataLoader(self.dsetValid, batch_size=batchSize,
                                        shuffle=True, num_workers=numWorkers,
                                        collate_fn=collate_wrapper,
                                        pin_memory=pinMemory
                                        )
