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
from utils import loadTrainedModel, retrieveDataloader, loadOTSModel

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
    n_edit_steps = 5, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):
    
    edit_tokens, edit_mask = edit_example
    edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
    edit_labels = edit_tokens.masked_fill(edit_mask == 0, -100) 
    
    model.train()
    model_ = copy.deepcopy(model)
    inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)
    print("starting edit")
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
        
        # for p_, fp in zip(model_.parameters(), fmodel.parameters()):
        #     p_[:] = fp[:]
        model_.load_state_dict(fmodel.state_dict())
    
    return model_

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
        rank = torch.sum(probs > probs[gold_token]) 

    return rank

def evalSingleEdits(model, dataloader, model_name):
    
    outcomes = []
    
    model.to(DEVICE)
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    with open(f"../eval/{filename}", "w") as f:
        f.write((
            "train_step,success,success_diff,"
            "new_rank,new_ppl,orig_rank,orig_ppl\n"
            ))
        for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            edit_tokens, edit_mask = edit_example
            ent_tokens = ent[0].squeeze()
            ent_tokens = ent_tokens[ent_tokens != 50256].squeeze()
            
            edit_start_loc = np.min(np.argwhere(
                np.in1d(
                    edit_tokens.numpy(), 
                    ent_tokens.numpy()
                    )
                ).squeeze())

            edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
            edit_labels = edit_tokens.masked_fill(edit_mask == 0, -100) 
            edit_labels.to(DEVICE)
            
            gold_token = ent_tokens.cpu()
            gold_token = gold_token.item() if not gold_token.size() else gold_token[0].item()

            orig_rank = getIndexedProbs(
                model, 
                edit_start_loc, 
                gold_token, 
                edit_tokens, 
                edit_mask, 
                edit_labels
                )
            orig_ppl = perplexity(model, dataloader)

            model_out = performOneEdit(model, edit_example)
                    
            new_rank = getIndexedProbs(
                model_out, 
                edit_start_loc, 
                gold_token, 
                edit_tokens, 
                edit_mask, 
                edit_labels
                )
            new_ppl = perplexity(model_out, dataloader)

            success = new_rank < orig_rank
            success_diff = orig_rank - new_rank

            run = (
                train_step, success, success_diff, 
                new_rank, new_ppl, orig_rank, orig_ppl
                )
            outcomes.append(run)
            form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
            writeStr = ",".join([form(x) for x in run])
            f.write(f"{writeStr}\n")
        
    success_pct = sum([x[1] for x in outcomes]) / len(outcomes)
    return success_pct, outcomes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--ppl', action='store_true')
    parser.add_argument('--edit', action='store_true')
    args = parser.parse_args()

    model, tokenizer = loadOTSModel()
    # model, tokenizer = loadTrainedModel(args.model_path)
    
    

    if args.ppl:
        dataloader = retrieveDataloader(tokenizer, bs=15, dataset='valid', max_obs=50)
        ppl = runPPL(model_ots, dataloader, tokenizer, modelpath="ots")
        # ppl = runPPL(model, dataloader, tokenizer, modelpath=args.model_path)
        print(ppl)
    
    if args.edit:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='valid', max_obs=20)
        success_pct, outcomes = evalSingleEdits(model, dataloader, args.model_path)
        print(f"Success Pct: {success_pct}")
        
