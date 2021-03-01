import argparse
import tqdm
import glob
import os
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
from utils import loadTrainedModel, retrieveDataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def perplexity(model, dataloader):
    model.to(DEVICE)
    max_length = model.config.n_positions
    stride = 512

    ppls = []
    for batch_idx, lm_data in enumerate(dataloader):
        lm_tokens, lm_mask = lm_data[0]
        # lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
        # lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)

        lls = []
        for i in range(0, lm_tokens.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, lm_tokens.size(1))
            trg_len = end_loc - i    # may be different from stride on last loop
            input_ids = lm_tokens[:,begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:,:-trg_len] = -100
            lm_mask = lm_mask.to(DEVICE)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=lm_mask, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        print(f"Batch {batch_idx}; PPL {ppl}")
        ppls.append(ppl)

    return np.mean(np.array(ppls)), ppls

def backupPerplexity(model, dataloader):
    with torch.no_grad():
        for batch_idx, lm_data in enumerate(dataloader):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
            loss = model(
                lm_tokens,
                attention_mask=lm_mask,
                labels=lm_labels).loss

    return math.exp(loss)


def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def runEval(model, dataloader, modelpath=None):
    
    if not os.path.exists("../eval"):
        os.mkdir("../eval")
    
    ppl, pplList = perplexity(model, dataloader)

    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"evaluation_{timestamp}{os.path.basename(modelpath)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write(ppl)
        f.write("\n")
        f.write("-"*50 + "\n")
        f.write("\n".join(pplList))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    args = parser.parse_args()

    model, tokenizer = loadTrainedModel(args.model_path)
    model.eval()
    dataloader = retrieveDataloader(tokenizer, bs=10, dataset='valid')

    runEval(model, dataloader, modelpath=args.model_path)