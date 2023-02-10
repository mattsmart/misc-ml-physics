import numpy as np
import torch
from torch.utils.data import DataLoader


class CustomBatch():
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or max_len. Unclear to me whether this is instantiated
    at every call to dataloader or if it's instantiated along with dataloader.
    For now, all the tensors exist on CPU and are later pushed to the GPU. Needs
    potentially to be changed.

    Note on dimensions (at least for transformer_torch.py expected usage)
        - vec is a torch tensor representing a sequence of tokens
        - it is generally 2-dimensional: e.g. 10 tokens, each token in integer
            ID space, then:
                -vec.shape would be (1, 10) - for transformer_torch.py TODO confirm
                -vec.shape would be (10, 1) - for transformer_aiayn.py TODO confirm,
                 and then standardize them to be same
        - TODO - it may instead be a 10x1 or a 1x10 object where the "content"
                is the token integer ID

    Common scenario:  TODO - compare this example with TODO above for torch model
        - we have several sentences with different length in one "batch",
           each token however has fixed dimension e.g. R^512
        - example:
            - sentence 1: (1, 10)
            - sentence 2: (1, 7)
            - sentence 3: (1, 14)
        - this function is used to pad a given sentence -- vec -- to a given max
            length, e.g.
            - padded sentence 2: (1, 14)
    """
    def __init__(self, data, dim=0, pad_value=0, stack_dim=1,
                 max_len_model=None, flag_padding_mask=True):
        """
        # TODO name changes and documentation
        Input:
            data (dataset)       : a batch of dataset.
            dim_sentence (int)   : the dimension to be padded (dimension of time in sequences)
                                    0 - huggingface, 1 - annotated transformer
            max_len_model (int)  : maximum length of sentence, if any
            pad_value (int)      : value of padding token (0 is generally used as the pad value)
            stack_dim (int)      : dimension along which to stack the data tensor.
                                    1 - huggingface, 0 - annotated transformer
        """
        self.dim = dim
        self.padValue = pad_value

        max_len_seq = np.max([x.shape[self.dim] for x in data])
        self.maxLen = np.min([x for x in [max_len_seq, max_len_model]
                              if x is not None])
        # pad according to max_len
        batch = [self.pad_tensor(x[:self.maxLen]) for x in data]
        # stack all, change to dim = 0 for annotated transformer?
        self.src = (torch.stack([x[:-1] for x in batch], dim=stack_dim)).long()
        self.tgt = (torch.stack([x[1:] for x in batch], dim=stack_dim)).long()
        '''key_padding_mask: :math:(N, S) where N is the batch size, S is
            the source sequence length. If provided, specified padding
            elements in the key will be ignored by the attention. This is an
            binary mask. When the value is True, the corresponding value on
            the attention layer will be filled with -inf'''
        if flag_padding_mask:
            src_pad_mask = (self.src == self.padValue)
            self.src_pad_mask = src_pad_mask.transpose(0, 1)
        else:
            self.src_pad_mask = None
        #  self.tgt_pad_mask = (self.tgt != self.pad_value).type(torch.int)
        #  ys = torch.LongTensor(map(lambda x: x[1], batch))

    def pad_tensor(self, vec):
        """
        Padding a torch tensor 'vec' to the max length in the batch.
        Input:
            vec : torch tensor which is to be padded to the size "pad_size_int"
        Output:
            a new tensor padded to 'pad_size_int' in dimension 'dim_sentence'
        """
        pad_size = list(vec.shape)
        pad_size[self.dim] = self.maxLen - vec.size(self.dim)
        return torch.cat([vec, self.padValue * torch.ones(*pad_size)], dim=self.dim)

    def pin_memory(self):
        # Usage: potential code speedup; intended for GPU usage
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


class CustomDataloader():
    """

    """
    def __init__(self, dataset, batch_size, max_len,
                 dim=0,
                 num_workers=2,
                 pin_memory=True,
                 split_train_test_val=(0.7, 0.2, 0.1),
                 flag_padding_mask=True):
        nbr_train = np.floor(split_train_test_val[0] * len(dataset))
        nbr_test = np.floor(split_train_test_val[1] * len(dataset))
        nbr_valid = len(dataset) - (nbr_train + nbr_test)
        split_train_test_val = [nbr_train, nbr_test, nbr_valid]
        self.dataset_train, self.dataset_test, self.dataset_valid = \
            torch.utils.data.random_split(
                dataset,
                [int(x) for x in split_train_test_val],
                generator=torch.Generator().manual_seed(42)
            )

        tknzr = dataset.transform

        def collate_wrapper(batch):
            return CustomBatch(batch, dim=dim, max_len_model=max_len,
                               pad_value=tknzr.get_vocab()["<pad>"],
                               flag_padding_mask=flag_padding_mask)

        self.train = DataLoader(self.dataset_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=collate_wrapper,
                                pin_memory=pin_memory
                                )
        self.test = DataLoader(self.dataset_test, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers,
                               collate_fn=collate_wrapper,
                               pin_memory=pin_memory
                               )
        self.valid = DataLoader(self.dataset_valid, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=collate_wrapper,
                                pin_memory=pin_memory
                                )
