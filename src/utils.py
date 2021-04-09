import os
import platform
import zipfile
from shutil import copyfile

import glob
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_process import TorchDataset

def loadOTSModel(cache_dir=None):
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def retrieveDataloader(
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


def loadTrainedModel(modelPath, cache_dir=None, tokenizer=True):
    model, tok = loadOTSModel(cache_dir=cache_dir)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    if not tokenizer:
        return model
    return model, tok

def locateEntityEdit(edit_tokens, ent_tokens):
    return np.argwhere(
        np.in1d(
            edit_tokens.numpy(), 
            ent_tokens.numpy()
            )
        ).flatten()


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
    copyfile(
        "/juice/scr/spencerb/editable_nlp/data.zip", 
        f"{local_dir}/data.zip"
        )
    with zipfile.ZipFile(f"{local_dir}/data.zip") as zf:
        zf.extractall(f"{local_dir}")

    
    return local_dir
