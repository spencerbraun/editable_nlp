import argparse
import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from data_process import TorchDataset
from utils import loadTrainedModel, retrieveDataloader, loadOTSModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def perplexity(model, dataloader):
    total_loss = []
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch_idx, (lm_data, edit_sample, ent) in enumerate(dataloader):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
            out = model(
                lm_tokens,
                # attention_mask=lm_mask,
                labels=lm_labels)

            loss = out.loss
            total_loss.append(loss)

    return torch.exp(torch.mean(torch.stack(total_loss)))


def runPPL(model, dataloader, modelpath="None"):
    
    if not os.path.exists("../eval"):
        os.mkdir("../eval")
    
    ppl = perplexity(model, dataloader)
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"ppl_{timestamp}_{os.path.basename(modelpath)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write(str(int(ppl.cpu().numpy())))


def performOneEdit(
    model, 
    edit_example,
    n_edit_steps = 10, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):
    
    model.train()
    inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)
    
    print("starting edit")

    edit_tokens, edit_mask = edit_example
    edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
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

            print(f"Edit step {edit_step}; loss {loss} ") 

        return fmodel

def getIndexedProbs(model, index, gold_token, sent_tokens, mask, labels):
       
    model.eval()
    with torch.no_grad():
        output = model(
            sent_tokens, 
            attention_mask=mask,
            labels=labels
        )
        probs = (
            output.logits[:,index,:]
            .softmax(dim=-1).detach().cpu().squeeze()
            )
        ranking = np.where(torch.argsort(probs).numpy() == gold_token)
        rank = ranking[0].item()

    return rank

def evalSingleEdits(model, dataloader, model_name):
    
    outcomes = []
    
    model.to(DEVICE)
    for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
        edit_tokens, edit_mask = edit_example
        edit_start_loc = np.min(np.argwhere(
            np.in1d(
                edit_tokens.numpy(), 
                ent[0].numpy()
                )
            ).squeeze())

        edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
        edit_labels = edit_mask.masked_fill(edit_mask == 0, -100) 
        edit_labels.to(DEVICE)
        
        gold_token = ent[0].cpu().squeeze()
        gold_token = gold_token.item() if not gold_token.size() else gold_token[0].item()
    

        orig_rank = getIndexedProbs(
            model, 
            edit_start_loc, 
            gold_token, 
            edit_tokens, 
            edit_mask, 
            edit_labels
            )

        model_out = performOneEdit(model, edit_example)
                
        new_rank = getIndexedProbs(
            model_out, 
            edit_start_loc, 
            gold_token, 
            edit_tokens, 
            edit_mask, 
            edit_labels
            )
        
        success = new_rank < orig_rank
        success_diff = orig_rank - new_rank

        outcomes.append((train_step, success, success_diff))

    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write("\n".join([f"{x[0]},{x[1]},{x[2]}" for x in outcomes]))
        f.write("\n")
    
    success_pct = sum([x[1] for x in outcomes]) / len(outcomes)
    return success_pct


# def score(sentence):
#     tokenize_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     loss=model(tensor_input, lm_labels=tensor_input)
#     return math.exp(loss)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument('--edit', action='store_true')
    args = parser.parse_args()

    model, tokenizer = loadTrainedModel(args.model_path)
    # model, tokenizer = loadOTSModel()
    model.eval()
    

    if args.ppl:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='valid', max_obs=50)
        runPPL(model, dataloader, modelpath=args.model_path)
    
    if args.edit:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='valid', max_obs=20)
        success_pct = evalSingleEdits(model, dataloader, args.model_path)
        print(f"Success Pct: {success_pct}"")
        