import tqdm
import glob
import time
import random
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset, list_metrics, load_metric

from data_process import TorchDataset
from utils import load_model 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def validDataloader(tokenizer, bs=10):
    writtenFiles = glob.glob("../data/valid/permuted*")
    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    dataset = TorchDataset(list(range(fileIndex)), tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs
    )

    return dataloader


def perplexity(model, dataloader):

    max_length = model.config.n_positions
    stride = 512

    
    for batch_idx, lm_data in enumerate(dataloader):
        lm_tokens, lm_mask = lm_data
        # lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
        # lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)

        lls = []
        for i in tqdm(range(0, lm_tokens.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, lm_tokens.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = lm_tokens[:,begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)


    # with torch.no_grad():
    #     for batch_idx, lm_data in enumerate(dataloader):
    #         lm_tokens, lm_mask = lm_data
    #         lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
    #         lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
    #         outputs = model(
    #             lm_tokens
    #             attention_mask=lm_mask,
    #             labels=lm_labels).loss

    #         ll = outputs[0] * trg_len

    #     lls.append(log_likelihood)

    #     ppl = torch.exp(torch.stack(lls).sum() / end_loc)

        return math.exp(loss)

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def runEval():

