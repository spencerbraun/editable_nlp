import os
import platform
import zipfile
from shutil import copyfile
import math

import glob
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
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
    def n_predictions_for_labels(labels, offset=1):
        if labels.dim() == 1:
            return (labels[offset:] != -100).sum().item()
        elif labels.dim() == 2:
            return (labels[:, offset:] != -100).sum().item()
        else:
            assert False




def loadOTSModel(name='gpt2', cache_dir=None):
    if name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            name, cache_dir=f"{cache_dir}/hf" if cache_dir else None
            )
        tokenizer = GPT2Tokenizer.from_pretrained(
            name, cache_dir=f"{cache_dir}/hf" if cache_dir else None
            )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif name == 't5-small':
        model = T5ForConditionalGeneration.from_pretrained(
        "t5-small", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
        tokenizer = T5Tokenizer.from_pretrained(
            't5-small', cache_dir=f"{cache_dir}/hf" if cache_dir else None
            )
    elif name == 'bart-base':
        model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-base", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
        tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )

    return model, tokenizer

def loadTokenizer(name='gpt2', cache_dir=None):
    if name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            name, cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    elif name == 't5-small':
        tokenizer = T5Tokenizer.from_pretrained(
            't5-small', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    elif name == 'bart-base':
        tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )

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


def wrap_model(model, name, ortho=False):
    # find Conv1D layers to replace (they annoyingly have transposed weights)
    modules = None
    if name == 'gpt2':
        d = 3072
        module_predicate = lambda mod: (
            isinstance(mod, transformers.models.gpt2.modeling_gpt2.Conv1D) and mod.weight.shape[1] == d
        )
        n_hidden = d #model.config.n_embd
        modules = None
    elif name == 'bart-base':
        module_predicate = lambda mod: (
            isinstance(mod, transformers.models.bart.modeling_bart.BartDecoderLayer) or
            isinstance(mod, transformers.models.bart.modeling_bart.BartEncoderLayer)
        )
        n_hidden = model.config.d_model
    elif name == 't5-small':
        module_predicate = lambda mod: (
            isinstance(mod, transformers.models.t5.modeling_t5.T5DenseReluDense)
        )
        n_hidden = model.config.d_model
    else:
        raise ValueError(f"Invalid model type `{name}` specified")

    ConditionalLinearWrapper.wrap_model(model, n_hidden, -1, module_predicate, ortho=ortho, modules=modules)

def prep_for_maml(model, adapt_all: bool = False):
    # Default inner loop adaptation parameters
    norm_pred = lambda n: 'ln' not in n and 'bn' not in n
    def _inner_params(self):
        if hasattr(self, 'transformer'): # gpt2
            if adapt_all:
                return [p for (n,p) in self.transformer.h.named_parameters() if norm_pred(n)]
            else:
                return [p for (n,p) in self.transformer.h[-3:].named_parameters() if norm_pred(n)]
        elif hasattr(self, 'model'): # bart
            return (list(self.model.encoder.layers[-3:].parameters()) + 
                    list(self.model.decoder.layers[-3:].parameters()))
        else: # t5
            return (list(self.encoder.block[-3:].parameters()) + 
                    list(self.decoder.block[-3:].parameters()))

    type(model).inner_params = _inner_params


def loadTrainedModel(
    modelPath=None, 
    name='gpt2', 
    cache_dir=None, 
    tokenizer=True, 
    split_params: bool = False, 
    adapt_all: bool = False,
    ortho: bool = False,
    ewc: bool = False,
    ):
    model, tok = loadOTSModel(name=name, cache_dir=cache_dir)
    prep_for_maml(model, adapt_all)
    if split_params:
        wrap_model(model, name, ortho=ortho)

    try:
        model.load_state_dict(torch.load(modelPath))
        print(f"Loaded checkpoint from {modelPath}")
    except Exception as e:
        print(f"Couldn't load pre-trained weights for model: {e}; continuing with OTS weights")
    
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
        local_dir = '/juice/scr/spencerb/results'

        if os.path.exists(local_dir) | debug:
            return local_dir

        os.mkdir(f"{local_dir}")
        os.mkdir(f"{local_dir}/models")
        os.mkdir(f"{local_dir}/models/finetune")
        os.mkdir(f"{local_dir}/errors")
        os.mkdir(f"{local_dir}/eval")
        os.mkdir(f"{local_dir}/hf")
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

