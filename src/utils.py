import os
import platform
import zipfile
from shutil import copyfile

import glob
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_process import TorchDataset, WikitextDataset

def loadOTSModel(cache_dir=None):
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def loadTokenizer(cache_dir=None):
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def retrieveEditDataloader(
    tokenizer, 
    bs=10, 
    data_loc='..',
    dataset='train', 
    max_obs=float('inf'),
    shuffle=False
    ):

    writtenFiles = (
        glob.glob(f"{data_loc}/data/permuted*") if dataset == 'train' 
        else glob.glob(f"{data_loc}/data/valid/original*") if dataset == 'valid' 
        else glob.glob(f"{data_loc}/data/test/original*")
        )

    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    limitIndex = min(max_obs, fileIndex)
    ds = TorchDataset(list(range(limitIndex)), tokenizer, data_loc=data_loc, dataset=dataset)
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        num_workers=2,
        pin_memory=True,
        shuffle=shuffle
    )

    return dataloader

def wikiDataloader(
    tokenizer, 
    bs=10, 
    data_loc='..',
    dataset='train',
    shuffle=False,
    max_length=200,
    min_length=20
    ):

    def pad_collate(batch):
        tokens = [x[0] for x in batch]
        mask = [x[1] for x in batch]
        tok_pad = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        mask_pad = torch.nn.utils.rnn.pad_sequence(
            mask, batch_first=True, 
            padding_value=0
        )
        return tok_pad, mask_pad

    ds = WikitextDataset(
        tokenizer, 
        data_loc=data_loc, 
        dataset=dataset,
        pct=100, 
        max_length=max_length,
        min_length=min_length
        )
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        num_workers=2,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=pad_collate if bs > 1 else None
    )

    return dataloader


def loadTrainedModel(modelPath, cache_dir=None, tokenizer=True):
    model, tok = loadOTSModel(cache_dir=cache_dir)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    if not tokenizer:
        return model
    return model, tok

def locateSubset(whole, subset):
    whole = whole.flatten().cpu().numpy()
    subset = subset.flatten().cpu().numpy()
    start = subset[0]
    len_sub = len(subset)
    locs = np.where(whole == start)[0]
    empty = torch.tensor([])

    for loc in list(locs):
        whole_slice = whole[loc:(loc+len_sub)]
        if len(subset) != len(whole_slice):
            return empty
        if np.all(subset == whole_slice):
            sub_loc = torch.tensor(range(loc,(loc+len_sub)))
            return sub_loc

    return empty


def sailPreprocess(debug=False):
    machine_name = platform.node().split(".")[0]
    scr = max(os.listdir(f"/{machine_name}"))
    save_loc = f"/{machine_name}/{scr}"
    local_dir = f"{save_loc}/spencerb"
    if os.path.exists(local_dir) | debug:
        return local_dir
    
    os.mkdir(f"{save_loc}/spencerb")
    os.mkdir(f"{save_loc}/spencerb/models")
    os.mkdir(f"{save_loc}/spencerb/models/finetune")
    os.mkdir(f"{save_loc}/spencerb/errors")
    os.mkdir(f"{save_loc}/spencerb/eval")
    os.mkdir(f"{save_loc}/spencerb/hf")
    
    return local_dir
