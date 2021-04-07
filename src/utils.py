import glob
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_process import TorchDataset

def loadOTSModel(cache_dir=None):
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir if cache_dir else None)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir if cache_dir else None)
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
    dataset = TorchDataset(list(range(limitIndex)), tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=4,
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