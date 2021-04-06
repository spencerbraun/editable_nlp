import random
import copy 
import argparse
import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

import utils
from data_process import TorchDataset
from utils import loadTrainedModel, retrieveDataloader, loadOTSModel, locateEntityEdit

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

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
               
        if len(list(gold_tokens.shape)) == 1:
            gold_tokens.unsqueeze_(1)
        logit_sum = torch.sum(logits.gather(1, gold_tokens))
        probs_avg = torch.mean(probs.gather(1, gold_tokens))
    return logit_sum, probs_avg

def performOneEdit(
    model, 
    edit_tokens,
    edit_mask,
    edit_labels,
    n_edit_steps=5,
    lr=1e-3
    ):
    
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
            if loss < 1 and edit_step > 0:
                break
            diffopt.step(loss)

            print(f"Edit step {edit_step}; loss {loss} ") 
        
        model_.load_state_dict(fmodel.state_dict())
    
    return model_

def genModelText(finetuned, lm_tokens, edit_locs):
        

        input_ids = lm_tokens[:, :edit_locs.min()]
        input_size = input_ids.size()[-1]
        
        finetuned.eval()
        print("generating")
        output_sequence = finetuned.generate(
            input_ids=input_ids,
            max_length=input_size + 5,
            temperature=0.7,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=10,
        )

        edit_tokens = random.choice(output_sequence).unsqueeze(0)
        edit_mask = torch.ones(edit_tokens.shape, dtype=torch.long)
        edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
        edit_labels[:, input_size:] = edit_tokens[:, input_size:]
        gold_tokens = edit_tokens[:, input_size:]

        edit_labels = edit_labels.to(DEVICE)
        edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)

        return edit_tokens, edit_mask, edit_labels, gold_tokens

def evalSequentialEdits(
    model, 
    dataloader, 
    model_name, 
    n_edit_steps,
    seq_edits=1,
    self_sample=False, 
    testset=False
    ):

    if self_sample:
        finetuned = utils.loadTrainedModel(
            "../models/finetune/gpt2_epoch0_ts10000.20210310.18.03.1615401990", 
            tokenizer=False
        )
        finetuned.to(DEVICE)
    
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    
    model.to(DEVICE)
    n_edits = 0
    edit_number = 1
    sequential = seq_edits > 1
    if sequential:
        model_edited = copy.deepcopy(model)

    saveloc = f"../eval/{filename}" if not testset else f"../eval/test/{filename}" 
    with open(saveloc, "w") as f:
        f.write((
            "train_step,n_edit_steps,edit_number,success,success_diff,"
            "new_logits,orig_logits,new_ppl,orig_ppl,new_prob,old_prob\n"
            ))
        for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            
            edit_tokens, edit_mask = edit_example
            ent_tokens = ent[0].flatten() #1d array of vocab indexes
            ent_tokens = ent_tokens[ent_tokens != 50256]
    
            edit_locs = locateEntityEdit(edit_tokens, ent_tokens)
            if edit_locs.size == 0 or (edit_locs.min() == 0 & self_sample):
                print(f"Skipping {train_step}")
                continue
            
            if self_sample:
                edit_tokens, edit_mask, edit_labels, gold_tokens = genModelText(
                    finetuned, lm_tokens, edit_locs
                    )
            
                gold_tokens = gold_tokens.cpu()
            else:
                edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
                edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
                edit_labels = edit_labels.to(DEVICE)
                edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
                
                gold_tokens = ent_tokens.cpu()
            
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
                model if not sequential else model_edited, 
                edit_tokens, 
                edit_mask, 
                edit_labels,
                n_edit_steps=n_edit_steps, 
                lr=1e-3
                )

            if (edit_number < n_edits) & sequential:
                edit_number += 1
                model_edited.load_state_dict(model_out.state_dict())
            else:
                edit_number = 1
                if sequential:
                    model_edited.load_state_dict(model.state_dict())
                    
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
                train_step, n_edit_steps, edit_number, success, success_diff, 
                new_logits, orig_logits, new_ppl, orig_ppl, new_prob, orig_prob
                )

            form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
            writeStr = ",".join([form(x) for x in run])
            f.write(f"{writeStr}\n")

            n_edits +=1 
            if n_edits >= (100 * seq_edits):
                break

class ModelComps:
    def __init__(self, model_name, base_name, archive=False, test=False):
        
        self.test = test
        self.archive = archive
        self.model_name = model_name
        self.base_name = base_name
        self.ots_name = "OTS"
        self.models = {}
        self.modelStats = self.getModelParams()
        self.stats = {}
    
    
    def getModelParams(self):
        model_id = ".".join(self.model_name.split(".")[1:])
        hyper_obj = torch.load(f"../models/hypers.{model_id}")
        if isinstance(hyper_obj, dict):
            return pd.Series(hyper_obj).to_frame().T
        else:
            hyper_dict = {
                "inner_lr": hyper_obj.inner_lr,
                "outer_lr": hyper_obj.outer_lr,
                "epochs": hyper_obj.epochs,
                "n_edit_steps": hyper_obj.n_edit_steps,
                "cedit": hyper_obj.cedit,
                "cloc": hyper_obj.cloc 
            }
            return pd.Series(hyper_dict).to_frame().T

    def readData(self, model_name, kind='model'):
        if self.archive:
            eval_glob = glob.glob(f"./archive/*{model_name}")
        elif self.test:
            eval_glob = glob.glob(f"./test/*{model_name}")
        else:
            eval_glob = glob.glob(f"./*{model_name}")
        for evaluation in eval_glob:
            df = pd.read_csv(evaluation)
            eval_id = f"{kind}_{evaluation.split('.')[4].split('_')[0]}"
            self.models[eval_id] = df
    
    def runStats(self):
        if not self.models:
            self.readData(self.base_name, kind='base')
            self.readData(self.model_name)
            self.readData(self.ots_name, kind='ots')
            
        for name, model in self.models.items():  
            mean_new_ppl = model.new_ppl.mean() 
            mean_orig_ppl = model.orig_ppl.mean() 
            pct_ppl_dd = model.apply(
                lambda x: 100*(x.new_ppl - x.orig_ppl)/x.orig_ppl, axis=1).mean()
            gross_ppl_dd = model.apply(
                lambda x: x.new_ppl - x.orig_ppl, axis=1).mean()
            new_logits_higher = model.success.mean() * 100
            new_logits_higher05 = model.apply(
                lambda x: x.new_logits > 0.95*x.orig_logits,axis=1).mean() * 100
            new_logits_higher10 = model.apply(
                lambda x: x.new_logits > 0.90*x.orig_logits,axis=1).mean() * 100

            success_by_probs = model.apply(lambda x: x.new_prob > x.old_prob, axis=1).mean()
            n_edit_steps = model.n_edit_steps.max()

            
            self.stats[name] = {
                "edit_steps":n_edit_steps,
                "mean_new_ppl":mean_new_ppl,
                "mean_orig_ppl":mean_orig_ppl,
                "pct_ppl_dd":pct_ppl_dd,
                "gross_ppl_dd":gross_ppl_dd,
                "new_logits_higher":new_logits_higher,
                "new_logits_higher05":new_logits_higher05,
                "new_logits_higher10":new_logits_higher10,
                "success_by_probs":success_by_probs,
            }
    
    @property
    def statDf(self):
        stats_df = (
            pd.DataFrame(self.stats).T
            .reset_index()
            .rename(columns={'index':'model'})
            .sort_values(["model", "edit_steps"], ascending=False)
        )
        
        return stats_df
    
    def summary(self):
        if not self.stats:
            self.runStats()
        print("Model Parameters:")
        display(self.modelStats)
        
        print("Success Metrics")
        stats_df = (
            pd.DataFrame(self.stats).T
            .reset_index()
            .rename(columns={'index':'model'})
            .sort_values(["model", "edit_steps"], ascending=False)
        )
        
        display(stats_df)
                
    def plotter(self, xlim=[]):
        print("Plotting Logits")
        for name, model in self.models.items():
            print(name)

            plt.hist(model.orig_logits, alpha=0.4, label="Pre-Edit")
            plt.hist(model.new_logits, alpha=0.4, label="Post-Edit")
            if name[:3] == 'ots':
                plt.legend(fontsize='large')
            
            if xlim:
                plt.xlim(xlim)
            plt.xlabel("Edited Entity Logits", fontsize='large')
            plt.ylabel("", fontsize='large')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--ots', action='store_true')
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--self_sample', action='store_true')
    parser.add_argument('--edit_steps', default=1)
    args = parser.parse_args()

    if args.model_path: 
        model, tokenizer = loadTrainedModel(args.model_path)
    else:
        model_ots, tokenizer = loadOTSModel()
    if args.test_set:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='test', max_obs=200)
    else:
        dataloader = retrieveDataloader(tokenizer, bs=1, dataset='valid', max_obs=100, shuffle=True)

    if args.model_path:
        evalSequentialEdits(
            model, 
            dataloader, 
            args.model_path, 
            int(args.edit_steps),
            self_sample=args.self_sample,
            testset=args.test_set
            )

    if args.ots:
        evalSequentialEdits(
            model_ots, 
            dataloader, 
            "OTS", 
            int(args.edit_steps),
            testset=args.test_set
            )