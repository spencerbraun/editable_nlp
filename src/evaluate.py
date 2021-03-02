import math
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

# def perplexity(model, dataloader):
#     model.to(DEVICE)
#     max_length = model.config.n_positions
#     stride = 512

#     ppls = []
#     for batch_idx, lm_data in enumerate(dataloader):
#         lm_tokens, lm_mask = lm_data[0]
#         # lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
#         # lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)

        
#         for i in range(0, lm_tokens.size(1), stride):
#             begin_loc = max(i + stride - max_length, 0)
#             end_loc = min(i + stride, lm_tokens.size(1))
#             trg_len = end_loc - i    # may be different from stride on last loop
#             input_ids = lm_tokens[:,begin_loc:end_loc].to(DEVICE)
#             target_ids = input_ids.clone()
#             target_ids[:,:-trg_len] = -100
#             lm_mask = lm_mask.to(DEVICE)

#             with torch.no_grad():
#                 outputs = model(input_ids, attention_mask=lm_mask, labels=target_ids)
#                 log_likelihood = outputs[0] * trg_len

#             lls.append(log_likelihood)
#         ppl = torch.exp(torch.stack(lls).sum() / end_loc)
#         print(f"Batch {batch_idx}; PPL {ppl}")
#         ppls.append(ppl)

#     return np.mean(np.array(ppls)), ppls

def backupPerplexity(model, dataloader):
    total_loss = []
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch_idx, (lm_data, _) in enumerate(dataloader):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
            out = model(
                lm_tokens,
                attention_mask=lm_mask,
                labels=lm_labels)

            loss = out.loss
            total_loss.append(loss)

    return torch.exp(torch.mean(torch.stack(total_loss)))



def runPPL(model, dataloader, modelpath=None):
    
    if not os.path.exists("../eval"):
        os.mkdir("../eval")
    
    ppl = backupPerplexity(model, dataloader)
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"evaluation_{timestamp}{os.path.basename(modelpath)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write(str(int(ppl.cpu().numpy())))


# def score(sentence):
#     tokenize_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     loss=model(tensor_input, lm_labels=tensor_input)
#     return math.exp(loss)


def performOneEdit(
    model, 
    lm_data, 
    edit_example,
    n_edit_steps = 10, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):
    
    model.train()
    inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)
    
    print("starting edit")

    lm_tokens, lm_mask = lm_data
    lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
    edit_tokens, edit_mask = edit_example
    edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
    
    lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
    edit_labels = edit_mask.masked_fill(edit_mask == 0, -100) 
    
    with higher.innerloop_ctx(
        model, 
        inner_opt, 
        copy_initial_weights=False, 
        track_higher_grads=False
        ) as (fmodel, diffopt):
        
        for edit_step in range(n_edit_steps):

            loss = fmodel(
                edit_tokens, 
                attention_mask=edit_mask,
                labels=edit_labels
            ).loss
            diffopt.step(loss)

        base_out = model(
            lm_tokens, 
            attention_mask=lm_mask,
            labels=lm_labels
        )
        l_base = base_out.loss
        
        edit_out = fmodel(
            edit_tokens, 
            attention_mask=edit_mask,
            labels=edit_labels
        )
        l_edit = edit_out.loss
        l_loc = F.kl_div(
            edit_out.logits,
            base_out.logits,
            reduction='batchmean',
            log_target=True
        )
        
        total_loss = l_base + cedit * l_edit + cloc * l_loc
        total_loss.backward()

        print((
            f"One edit: ",
            f"L_edit {l_edit} L_base {l_base} L_loc {l_loc}; ",
            f"Total Loss {total_loss}"
        )) 

    return model

def evalSingleEdits(model, dataloader):

    losses = []
    drawdown = []
    model.to(DEVICE)
    for train_step, (lm_data, edit_example) in enumerate(dataloader):
        edit_tokens, edit_mask = edit_example
        edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
        edit_labels = edit_mask.masked_fill(edit_mask == 0, -100) 
        edit_labels.to(DEVICE)
        model.eval()

        loss = model(
            edit_tokens, 
            attention_mask=edit_mask,
            labels=edit_labels
        ).loss


        model_out = performOneEdit(model, lm_data, edit_example)
        model_out.eval()

        edit_loss = model_out(
            edit_tokens, 
            attention_mask=edit_mask,
            labels=edit_labels
        ).loss

        drawdown.append(torch.exp(loss) - torch.exp(edit_loss))
        losses.append(loss - edit_loss)

    return drawdown, losses




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument('--edit', action='store_true')
    args = parser.parse_args()

    model, tokenizer = loadTrainedModel(args.model_path)
    model.eval()
    

    if args.ppl:
        dataloader = retrieveDataloader(tokenizer, bs=10, dataset='valid', max_obs=50)
        runPPL(model, dataloader, modelpath=args.model_path)
    
    if args.edit:
        dataloader = retrieveDataloader(tokenizer, bs=2, dataset='valid', max_obs=50)
        drawdown, losses = evalSingleEdits(model, dataloader)
        import ipdb; ipdb.set_trace()