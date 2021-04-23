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
    tokenizer.padding_side = "left"

    return model, tokenizer

def loadTokenizer(cache_dir=None):
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    
    def pad_collate(batch):
        toks = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True
            )
        return (
            torch.tensor(toks['input_ids']), 
            torch.tensor(toks['attention_mask'])
            )

    ds = WikitextDataset(
        data_loc=f"{data_loc}/hf", 
        dataset=dataset,
        pct=100, 
        min_length=min_length
        )
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        num_workers=2,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=pad_collate
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
    user = os.environ["USER"]
    if user == "spencerb":
        machine_name = platform.node().split(".")[0]
        scr = max(os.listdir(f"/{machine_name}"))
        save_loc = f"/{machine_name}/{scr}"
        local_dir = f"{save_loc}/{user}"

        if os.path.exists(local_dir) | debug:
            return local_dir

        os.mkdir(f"{save_loc}/{user}")
        os.mkdir(f"{save_loc}/{user}/models")
        os.mkdir(f"{save_loc}/{user}/models/finetune")
        os.mkdir(f"{save_loc}/{user}/errors")
        os.mkdir(f"{save_loc}/{user}/eval")
        os.mkdir(f"{save_loc}/{user}/hf")

    else:
        save_loc = "/iris/u"
        local_dir = f"{save_loc}/{user}/code/editable_nlp"

    return local_dir

