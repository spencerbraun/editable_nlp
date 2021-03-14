import copy 
import argparse
import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from data_process import TorchDataset
from utils import loadTrainedModel, retrieveDataloader, loadOTSModel, locateEntityEdit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def perplexity(model, dataloader):
    total_loss = []
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch_idx, (lm_data, _, _) in enumerate(dataloader):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            out = model(lm_tokens, labels=lm_labels)

            loss = out.loss
            total_loss.append(loss)

    return torch.exp(torch.mean(torch.stack(total_loss)))


def runPPL(model, dataloader, modelpath="None"):
    
    if not os.path.exists("../eval"):
        os.mkdir("../eval")
    
    ppl = perplexity(model, dataloader, tokenizer)
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"ppl_{timestamp}_{os.path.basename(modelpath)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write(str(int(ppl.cpu().numpy())))

    return ppl


def performOneEdit(
    model, 
    edit_example,
    edit_locs,
    n_edit_steps = 5,
    lr=1e-3
    ):
    
    edit_tokens, edit_mask = edit_example
    edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
    edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
    edit_labels = edit_labels.to(DEVICE)
    edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
    
    model.train()
    model_ = copy.deepcopy(model)
    inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)
    print("starting edit")
    with higher.innerloop_ctx(
        model, 
        inner_opt, 
        copy_initial_weights=False, 
        track_higher_grads=True
        ) as (fmodel, diffopt):
        
        for edit_step in range(n_edit_steps):

            loss = fmodel(
                edit_tokens, 
                attention_mask=edit_mask,
                labels=edit_labels
            ).loss
            if loss < 1:
                break
            diffopt.step(loss)

            print(f"Edit step {edit_step}; loss {loss} ") 
        
        # for p_, fp in zip(model_.parameters(), fmodel.parameters()):
        #     p_[:] = fp[:]
        model_.load_state_dict(fmodel.state_dict())
    
    return model_

def getIndexedProbs(model, index, gold_tokens, sent_tokens, mask, labels):
       
    model.eval()
    with torch.no_grad():
        output = model(
            sent_tokens, 
            attention_mask=mask,
            labels=labels
        )
        logits = output.logits[:,index,:].detach().cpu().squeeze(0) #squeeze batch size
        probs = (
            output.logits[:,index,:]
            .softmax(dim=-1).detach().cpu().squeeze(0)
            )
        # rank = torch.sum(probs > probs[gold_token]) 
        logit_sum = torch.sum(logits.gather(1, gold_tokens.unsqueeze(1)))
        probs_avg = torch.mean(probs.gather(1, gold_tokens.unsqueeze(1)))
    return logit_sum, probs_avg

def evalSingleEdits(model, dataloader, model_name, n_edit_steps):
    
    outcomes = []
    
    model.to(DEVICE)
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    n_edits = 0 
    with open(f"../eval/{filename}", "w") as f:
        f.write((
            "train_step,n_edit_steps,success,success_diff,"
            "new_logits,orig_logits,new_ppl,orig_ppl,new_prob,old_prob\n"
            ))
        for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            edit_tokens, edit_mask = edit_example
            ent_tokens = ent[0].flatten() #1d array of vocab indexes
            ent_tokens = ent_tokens[ent_tokens != 50256]
    
            try:
                edit_locs = locateEntityEdit(edit_tokens, ent_tokens)
            except:
                print(f"Error on step {train_step}, continuing")
                continue

            edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
            edit_labels = edit_tokens.masked_fill(edit_mask == 0, -100) 
            edit_labels.to(DEVICE)
            
            gold_tokens = ent_tokens.cpu()
            # gold_token = gold_token.item() if not gold_token.size() else gold_token[0].item()
            
            orig_logits, orig_prob = getIndexedProbs(
                model, 
                edit_locs, 
                gold_tokens, 
                edit_tokens, 
                edit_mask, 
                edit_labels
                )
            orig_ppl = perplexity(model, dataloader)

            model_out = performOneEdit(
                model, 
                edit_example, 
                edit_locs,
                n_edit_steps=n_edit_steps, 
                lr=1e-3
                )
                    
            new_logits, new_prob = getIndexedProbs(
                model_out, 
                edit_locs, 
                gold_tokens, 
                edit_tokens, 
                edit_mask, 
                edit_labels
                )
            new_ppl = perplexity(model_out, dataloader)

            success = new_logits > orig_logits
            success_diff = orig_logits - new_logits

            run = (
                train_step, n_edit_steps, success, success_diff, 
                new_logits, orig_logits, new_ppl, orig_ppl, new_prob, orig_prob
                )
            outcomes.append(run)
            form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
            writeStr = ",".join([form(x) for x in run])
            f.write(f"{writeStr}\n")

            n_edits +=1 
            if n_edits >= 50:
                break
        
    success_pct = sum([x[1] for x in outcomes]) / len(outcomes)
    return success_pct, outcomes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--ots', action='store_true')
    parser.add_argument('--edit', action='store_true')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--edit_steps', default=1)
    args = parser.parse_args()

    if args.model_path: 
        model, tokenizer = loadTrainedModel(args.model_path)
    else:
        model_ots, tokenizer = loadOTSModel()
    if args.test_set:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='test')
    else:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='valid', max_obs=100)
    
    
    if args.edit:
        success_pct, outcomes = evalSingleEdits(
            model, 
            dataloader, 
            args.model_path, 
            int(args.edit_steps)
            )
        
        print(f"Success Pct Trained: {success_pct}")

    if args.ots:
        success_pct_ots, outcomes = evalSingleEdits(
            model_ots, 
            dataloader, 
            "OTS", 
            int(args.edit_steps)
            )
        print(f"Success Pct OTS: {success_pct_ots}\n")
        
