import random
import copy 
import argparse
import os
from datetime import datetime

import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

import utils
from data_process import TorchDataset

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def perplexity(model, dataloader):
    total_loss = []
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for batch_idx, lm_data in enumerate(dataloader):
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

def loadLr(model_path):
    model_name = os.path.basename(model_path)
    model_id = model_name.split(".")[-1]
    step = model_name.split("_")[-1].split(".")[0]
    dir_loc = os.path.dirname(model_path)
    lr_glob = glob.glob(f"{dir_loc}/lr_epoch0_{step}.*{model_id}")

    if len(lr_glob) > 1:
        raise AttributeError("Too many lr specifications", ",".join(lr_glob))
    elif len(lr_glob) == 0:
        raise AttributeError("No lr specifications found")
    else:
        lrs = torch.load(lr_glob[0])

    return lrs

def performOneEdit(
    model, 
    lrs,
    edit_tokens,
    edit_mask,
    edit_labels,
    n_edit_steps=5,
    lr=1e-3
    ):
    
    model.train()
    model_ = copy.deepcopy(model)
    param_groups = [
        {'params': p, 'lr': None} 
        for p in model.transformer.h[-3:].parameters()
    ]
    inner_opt = (torch.optim.SGD(param_groups))
    
    print("starting edit")
    with higher.innerloop_ctx(
        model, 
        inner_opt, 
        override={'lr': lrs},
        copy_initial_weights=False, 
        track_higher_grads=True
        ) as (fmodel, diffopt):
        
        for edit_step in range(n_edit_steps):

            loss = fmodel(
                edit_tokens, 
                attention_mask=edit_mask,
                labels=edit_labels
            ).loss
            if loss == 0 and edit_step > 0:
                break
            diffopt.step(loss)

            print(f"Edit step {edit_step}; loss {loss} ") 
        
        model_.load_state_dict(fmodel.state_dict())
    
    return model_

def genModelText(finetuned, lm_tokens):
        
        lm_tokens = lm_tokens[lm_tokens != 50256]
        len_lm = lm_tokens.shape[-1]
        edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
        input_ids = lm_tokens[:edit_loc]
        input_size = input_ids.size()[-1]
        
        finetuned.eval()
        print(f"generating, {DEVICE}")
        output_sequence = finetuned.generate(
            input_ids=input_ids.unsqueeze(0),
            max_length=input_size + 5,
            temperature=1.2,
            do_sample=True,
            repetition_penalty=5.0,
            num_return_sequences=10,
        )

        edit_tokens = random.choice(output_sequence).unsqueeze(0)
        edit_mask = torch.ones(edit_tokens.shape, dtype=torch.long)
        edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
        edit_labels[:, input_size:] = edit_tokens[:, input_size:]
        gold_tokens = edit_tokens[:, input_size:]

        edit_labels = edit_labels.to(DEVICE)
        edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
        edit_locs = torch.tensor([edit_loc + i for i in range(5)])

        return edit_tokens, edit_mask, edit_labels, gold_tokens, edit_locs

def evalEditable(
    model, 
    dataloader, 
    model_name, 
    n_edit_steps,
    loc="..",
    testset=False
    ):
    
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    
    model.to(DEVICE)
    n_edits = 0
    saveloc = f"{loc}/eval/{filename}" if not testset else f"{loc}/eval/test/{filename}" 
    
    lrs = loadLr(model_name)

    with open(saveloc, "w") as f:
        f.write((
            "train_step,n_edit_steps,success,success_diff,"
            "new_logits,orig_logits,new_ppl,orig_ppl,new_prob,old_prob\n"
            ))
        for train_step, (lm_data, edit_example, new_ent, old_ent) in enumerate(dataloader):
            print(f"TS {train_step}")
            
            lm_tokens, lm_mask = lm_data
            orig_ent_tokens = old_ent[0].flatten()
            orig_ent_tokens = orig_ent_tokens[orig_ent_tokens != 50256]

            edit_tokens, edit_mask = edit_example
            ent_tokens = new_ent[0].flatten() #1d array of vocab indexes
            ent_tokens = ent_tokens[ent_tokens != 50256]

            edit_locs = utils.locateSubset(edit_tokens, ent_tokens)
            
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            
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
                model,
                lrs,
                edit_tokens, 
                edit_mask, 
                edit_labels,
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

            form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
            writeStr = ",".join([form(x) for x in run])
            f.write(f"{writeStr}\n")

            n_edits +=1 
            if n_edits >= 100:
                break



def evalSelfSample(
    model, 
    dataloader, 
    model_name, 
    n_edit_steps,
    loc="..",
    testset=False
    ):

    finetuned = utils.loadTrainedModel(
        f"{loc}/models/finetune/gpt2_epoch0_ts10000.20210408.09.04.1617899457", 
        cache_dir=loc,
        tokenizer=False
    )
    finetuned.to(DEVICE)
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    
    model.to(DEVICE)
    n_edits = 0
    saveloc = f"{loc}/eval/{filename}" if not testset else f"{loc}/eval/test/{filename}" 
    
    lrs = loadLr(model_name)

    with open(saveloc, "w") as f:
        f.write((
            "train_step,n_edit_steps,success,success_diff,"
            "new_logits,orig_logits,new_ppl,orig_ppl,new_prob,old_prob\n"
            ))
        for train_step, lm_data in enumerate(dataloader):
            print(f"TS {train_step}")
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

            edit_tokens, edit_mask, edit_labels, gold_tokens, edit_locs = genModelText(
                finetuned, lm_tokens
                )
                
            gold_tokens = gold_tokens.cpu()

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
                lrs,
                edit_tokens, 
                edit_mask, 
                edit_labels,
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

            form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
            writeStr = ",".join([form(x) for x in run])
            f.write(f"{writeStr}\n")

            n_edits +=1 
            if n_edits >= 100:
                break


class ModelComps:
    def __init__(self, model_name, base_name, loc="..", archive=False, test=False):
        
        self.test = test
        self.archive = archive
        self.model_name = model_name
        self.base_name = base_name
        self.loc = loc
        self.ots_name = "OTS"
        self.models = {}
        self.modelStats = self.getModelParams()
        self.stats = {}
        self.globs = []
    
    
    def getModelParams(self):
        model_id = ".".join(self.model_name.split(".")[1:])
        hyper_obj = torch.load(f"{self.loc}/models/hypers.{model_id}")
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
            eval_glob = glob.glob(f"{self.loc}/eval/archive/*{model_name}")
        elif self.test:
            eval_glob = glob.glob(f"{self.loc}/eval/test/*{model_name}")
        else:
            eval_glob = glob.glob(f"{self.loc}/eval/*{model_name}")
        for evaluation in eval_glob:
            self.globs.append(evaluation)
            df = pd.read_csv(evaluation)
            eval_id = f"{kind}_{evaluation.split('.')[5].split('_')[0]}"
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
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--self_sample', action='store_true')
    parser.add_argument('--edit_steps', default=1)
    args = parser.parse_args()

    loc = utils.sailPreprocess()

    if args.model_path: 
        model, tokenizer = utils.loadTrainedModel(args.model_path, cache_dir=loc)

    ds = 'test' if args.test_set else 'validation'
    if args.self_sample:
        dataloader = utils.wikiDataloader(tokenizer, bs=1, data_loc=loc, dataset=ds)
    else:
        dataloader = utils.retrieveEditDataloader(tokenizer, bs=1, data_loc=loc, dataset=ds)

    if args.self_sample:
        evalSelfSample(
            model, 
            dataloader,
            args.model_path, 
            int(args.edit_steps),
            loc=loc,
            testset=args.test_set
            )
    else:
         evalEditable(
            model,
            dataloader,
            args.model_path,
            int(args.edit_steps),
            loc=loc,
            testset=args.test_set
            )
