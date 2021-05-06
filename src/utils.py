import os
import platform
import zipfile
from shutil import copyfile
import math

import glob
import numpy as np
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_process import TorchDataset, WikitextDataset, NTokenDataset

from alg.senn_conditional import ConditionalLinearWrapper


class NLLAccumulator(object):
    def __init__(self):
        self.n_total = 0
        self.nll_sum = 0

    def update(self, nll: float, n: int):
        self.nll_sum += nll * n
        self.n_total += n

    def get_metrics(self):
        avg = self.nll_sum / self.n_total
        return avg, math.e ** avg

    @staticmethod
    def n_predictions_for_labels(labels):
        if labels.dim() == 1:
            return (labels[1:] != -100).sum().item()
        elif labels.dim() == 2:
            return (labels[:, 1:] != -100).sum().item()
        else:
            assert False


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


def _getFileIds(data_loc, self_sample, dataset, max_obs):
    data_path = f"{data_loc}/data/self_sample" if self_sample else f"{data_loc}/data"
    if self_sample:
        writtenFiles = (
            glob.glob(f"{data_path}/train/*") if dataset == 'train'
            else glob.glob(f"{data_path}/valid/*") if dataset == 'validation'
            else glob.glob(f"{data_path}/test/*")
        )
    else:
        writtenFiles = (
            glob.glob(f"{data_path}/train/permuted*") if dataset == 'train' 
            else glob.glob(f"{data_path}/valid/original*") if dataset == 'validation' 
            else glob.glob(f"{data_path}/test/original*")
        )

    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    limitIndex = min(max_obs, fileIndex)

    return list(range(limitIndex))


def retrieveEditDataloader(
    tokenizer, 
    bs=10, 
    data_loc='..',
    dataset='train', 
    max_obs=float('inf'),
    shuffle=False,
    self_sample=False,
    n_edits=1,
):

    ds = TorchDataset(
        _getFileIds(data_loc, self_sample, dataset, max_obs),
        tokenizer,
        data_loc=data_loc,
        dataset=dataset,
        self_sample=self_sample,
        n_edits=n_edits
    )
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        num_workers=2,
        pin_memory=True,
        shuffle=shuffle
    )

    return dataloader


def retrieveUnifiedDataset(
    tokenizer, 
    bs=10, 
    data_loc='..',
    dataset='train', 
    max_obs=float('inf'),
    shuffle=False,
    self_sample=False,
    n_edits=1,
):
    return NTokenDataset(
        _getFileIds(data_loc, self_sample, dataset, max_obs),
        tokenizer,
        data_loc=data_loc,
        dataset=dataset,
        self_sample=self_sample,
        batch_size=bs,
        n_edits=n_edits
    )


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


def split_conv_layers(model):
    # find Conv1D layers to replace (they annoyingly have transposed weights)
    conv_predicate = lambda mod: (
        isinstance(mod, transformers.models.gpt2.modeling_gpt2.Conv1D) and mod.weight.shape[1] == 768
    )
    ConditionalLinearWrapper.wrap_model(model, model.config.n_embd, -1, conv_predicate)


def prep_for_maml(model, adapt_all: bool = False):
    # Default inner loop adaptation parameters
    def _inner_params(self):
        if adapt_all:
            return list(self.transformer.h.parameters())
        else:
            return list(self.transformer.h[-3:].parameters())
    type(model).inner_params = _inner_params


def loadTrainedModel(modelPath, cache_dir=None, tokenizer=True, split_params: bool = False, adapt_all: bool = False):
    model, tok = loadOTSModel(cache_dir=cache_dir)
    prep_for_maml(model, adapt_all)
    if split_params:
        split_conv_layers(model)
        
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
        copyfile(
            "/juice/scr/spencerb/editable_nlp/self_sample.zip", 
            f"{local_dir}/self_sample.zip"
            )
        with zipfile.ZipFile(f"{local_dir}/self_sample.zip") as zf:
            zf.extractall(f"{local_dir}")
    else:
        save_loc = "/iris/u"
        local_dir = f"{save_loc}/{user}/code/editable_nlp"

    return local_dir

