import random
import copy
import argparse
import os
import re
import shutil
from datetime import datetime

import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from torch.utils.data import Subset

import utils
from data_process import TorchDataset
from masked_lm.data import MaskedLMDataloader

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def processLAMABatch(tokens, mask, label):
    return (tokens.to(DEVICE), mask.to(DEVICE),
            label.masked_fill(label == 0, -100).to(DEVICE))

def strip_padding(tokens, mask, labels):
    mask_ = tokens != 0
    tokens = tokens[mask_].unsqueeze(0)
    mask = mask[mask_].unsqueeze(0)

    label_mask = labels != 0
    labels = labels[label_mask].unsqueeze(0)

    return tokens, mask, labels


def perplexity(model, dataloader, lama=False, iteration=0):
    model.to(DEVICE)
    model.eval()
    acc = utils.NLLAccumulator()

    indices = np.random.default_rng(iteration).choice(len(dataloader), 200, replace=False)
    subset = Subset(dataloader, indices)
    dataset = dataloader if not lama else subset
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset):
            if lama:
                lm_tokens, lm_mask, lm_labels = processLAMABatch(
                    batch[0], batch[1], batch[2]
                    )
                loss = model(lm_tokens, labels=lm_labels).loss
                acc.update(loss.item(), acc.n_predictions_for_labels(lm_labels, offset=0))
            else:
                lm_data = batch[0]
                lm_tokens, lm_mask = lm_data[0]
                lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                loss = model(lm_tokens, labels=lm_labels).loss
                acc.update(loss.item(), acc.n_predictions_for_labels(lm_labels))

    avg_loss, ppl = acc.get_metrics()
    return torch.tensor(ppl)

def getIndexedProbs(model, index, gold_tokens, sent_tokens, labels):

    model.eval()
    with torch.no_grad():
        output = model(sent_tokens)
        output_lps = F.log_softmax(output.logits, dim=-1)
        logits = output_lps[:,index-1,:].detach().cpu().squeeze(0) #squeeze batch size

        gold_tokens = gold_tokens.flatten().unsqueeze_(1)
        logit_sum = torch.sum(logits.gather(1, gold_tokens))
        accuracy = (logits.argmax(-1) == gold_tokens.squeeze()).all(-1).float().mean()

    return (logit_sum, accuracy)

def getT5IndexedProbs(model, index, gold_tokens, sent_tokens, labels):

    model.eval()
    with torch.no_grad():
        output = model(input_ids=sent_tokens, labels=labels)
        output_lps = F.log_softmax(output.logits, dim=-1)
        logits = output_lps.detach().cpu().squeeze(0) #squeeze batch size

        gold_tokens = gold_tokens.flatten().unsqueeze_(1).cpu()
        logit_sum = torch.sum(logits.gather(1, gold_tokens))
        accuracy = (output_lps.argmax(-1) == labels).all(-1).float().mean().cpu()

    model.train()
    return (logit_sum, accuracy)

def loadLr(model_path):
    model_name = os.path.basename(model_path)
    model_id = model_name.split(".")[-1]
    step = re.search('ts.*\.', model_name).group(0)[:-1]
    split = '_split' if 'split' in model_name else ''
    dir_loc = os.path.dirname(model_path)
    lr_glob = glob.glob(f"{dir_loc}/lr{split}_epoch0_{step}.{model_id}")

    if len(lr_glob) > 1:
        raise AttributeError("Too many lr specifications", ",".join(lr_glob))
    elif len(lr_glob) == 0:
        raise AttributeError("No lr specifications found")
    else:
        print(f"Loading lrs {lr_glob[0]}")
        lrs = torch.load(lr_glob[0])

    return lrs

def performOneEdit(
    model, 
    task,
    lrs,
    edit_package,
    edit_locs,
    gold_tokens,
    n_edit_steps=10
    ):

    if not hasattr(model, "inner_params"):
        raise RuntimeError("Model has no `inner_params.` An `inner_params`"
                           " function should be patched in after model creation"
                           " and before editing.")
    model.train()
    param_groups = [
        {'params': p, 'lr': 1e-3}
        for p in model.inner_params()
    ]
    inner_opt = (torch.optim.SGD(param_groups))

    if task == 'gen':
        edit_tokens, edit_mask, edit_labels = edit_package
        idxProbs = getIndexedProbs
    elif task == 'cloze':
        (
            edit_tokens, edit_mask, edit_labels,
            edit_template, edit_temp_mask, edit_labels
            ) = edit_package

        idxProbs = getT5IndexedProbs

    logit_hist = []
    logit_hist.append(
        idxProbs(model, edit_locs, gold_tokens, edit_tokens,
        None if task == 'gen' else edit_labels),
    )

    with higher.innerloop_ctx(
        model,
        inner_opt,
        override={'lr': lrs} if lrs else None,
        copy_initial_weights=False,
        track_higher_grads=False # We don't need these because we're only evaluating (no outer loop)
        ) as (fmodel, diffopt):

        for edit_step in range(n_edit_steps):

            if hasattr(fmodel, "set_editing"):
                fmodel.set_editing(True)
            output = fmodel(
                edit_template,
                attention_mask=edit_temp_mask,
                labels=edit_labels
            )
            if hasattr(fmodel, "set_editing"):
                fmodel.set_editing(False)
            diffopt.step(output.loss)

            logit_hist.append(
                idxProbs(fmodel, edit_locs, gold_tokens, edit_tokens,
                None if task == 'gen' else edit_labels),
            )

        prob_change = logit_hist[-1][0].exp() - logit_hist[0][0].exp()
        print(f"prob history: {[l[0].exp() for l in logit_hist]}")
        print(f"Edit step {edit_step}; prob change {prob_change}; logit {logit_hist[-1][0]}; loss {output.loss}")

    edited_model = copy.deepcopy(model)
    edited_model.load_state_dict(fmodel.state_dict())

    return edited_model, logit_hist, prob_change, output.loss

def genModelText(finetuned, lm_tokens):

    len_lm = lm_tokens.shape[-1]
    edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
    input_ids = lm_tokens[:, :edit_loc]
    input_size = input_ids.size()[-1]

    finetuned.eval()
    print(f"generating, {DEVICE}")
    output_sequence = finetuned.generate(
        input_ids=input_ids,
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
    edit_locs = torch.tensor([edit_loc + i - 1 for i in range(5)])

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

    try:
        lrs = loadLr(model_name)
    except AttributeError:
        lrs = []

    with open(saveloc, "w") as f:
        f.write("train_step,n_edit_steps,logits,orig_ppl,new_ppl\n")
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

            orig_ppl = perplexity(model, dataloader)

            model_out, logit_hist, ll_change, loss = performOneEdit(
                model,
                lrs,
                edit_tokens,
                edit_mask,
                edit_labels,
                edit_locs,
                gold_tokens,
                n_edit_steps=n_edit_steps,
                lr=1e-3
            )

            new_ppl = perplexity(model_out, dataloader)

            for (val, acc) in logit_hist:
                run = (train_step, n_edit_steps, val, orig_ppl, new_ppl)
                form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
                writeStr = ",".join([form(x) for x in run])
                f.write(f"{writeStr}\n")

            n_edits +=1
            if n_edits >= 100:
                break


def processBatch(batch, lama=False):
    if lama:
        (
            _, _, _,
            _, _, _,
            edit_tokens, edit_mask, edit_labels,
            edit_template, edit_temp_mask, edit_labels
        ) = batch

        edit_template = edit_template[:1]
        edit_temp_mask = edit_temp_mask[:1]

        edit_tokens, edit_mask, _ = strip_padding(edit_tokens, edit_mask, edit_labels)
        edit_template, edit_temp_mask, edit_labels = strip_padding(edit_template, edit_temp_mask, edit_labels)

        edit_locs, gold_tokens = 0, edit_labels.flatten().unsqueeze(0).cpu()

        edit_package = (
            edit_tokens.to(DEVICE), edit_mask.to(DEVICE), edit_labels.to(DEVICE),
            edit_template.to(DEVICE), edit_temp_mask.to(DEVICE), edit_labels.to(DEVICE)
        )
    else:
        (lm_data, edit_example, _, _) = batch
        lm_tokens, lm_mask = lm_data[0]
        lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
        lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

        edit_tokens, edit_mask = edit_example[0]
        # remove left padding
        edit_tokens = edit_tokens.squeeze(0)
        edit_tokens = edit_tokens[edit_tokens != 50256].unsqueeze(0)
        edit_mask = edit_mask.squeeze(0)
        edit_mask = edit_mask[edit_mask != 0].unsqueeze(0)

        edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
        edit_loc = edit_tokens.shape[-1] - 5 - 1  # minus 1 for newline token
        edit_locs = torch.tensor([edit_loc + i for i in range(5)])
        edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
        gold_tokens = edit_tokens[:, edit_locs]

        edit_labels = edit_labels.to(DEVICE)
        edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)

        gold_tokens = gold_tokens.cpu()
        edit_package = (edit_tokens, edit_mask, edit_labels)

    return edit_package, edit_locs, gold_tokens


def evalSelfSample(
    model,
    dataloader,
    model_name,
    n_edit_steps,
    seq_edits=1,
    loc="..",
    testset=False,
    copy_to="",
    lama=False
    ):

    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    filename = f"edit_success_{timestamp}_{os.path.basename(model_name)}"
    saveloc = f"{loc}/eval/{filename}" if not testset else f"{loc}/eval/test/{filename}"

    try:
        lrs = loadLr(model_name)
    except AttributeError:
        print("No learning rates found!")
        lrs = []

    n_edits = 0
    edit_number = 0
    model_number = 0
    sequential = seq_edits > 1

    model_edited = copy.deepcopy(model).to(DEVICE)

    orig_ppl = perplexity(model, dataloader, lama=lama)
    orig_params = get_params(model)

    with open(saveloc, "w") as f:
        f.write(
            "model_number,edit_number,train_step,n_edit_steps,edit_step,"
            "logits,orig_ppl,new_ppl,norm\n"
            )
        for train_step, batch in enumerate(dataloader):
            print(f"Val Step {train_step}")
            print(f"Edit number {edit_number}")
            print(f"Model number {model_number}")
            edit_package, edit_locs, gold_tokens = processBatch(batch, lama=lama)

            model_edited, logit_hist, ll_change, loss = performOneEdit(
                model_edited,
                'cloze' if lama else 'gen',
                lrs,
                edit_package,
                edit_locs,
                gold_tokens,
                n_edit_steps=n_edit_steps
            )

            if (edit_number + 1) % 20 == 0:
                new_ppl = perplexity(model_edited, dataloader, lama=lama, iteration=n_edits)
            else:
                new_ppl = ""

            norm_diff = orig_params.sub(get_params(model_edited)).norm().item()

            for idx, (val, acc) in enumerate(logit_hist):
                run = (
                    model_number,edit_number,train_step, n_edit_steps, idx, val,
                    orig_ppl,new_ppl, norm_diff
                    )
                form = lambda x: str(x.cpu().item()) if torch.is_tensor(x) else str(x)
                writeStr = ",".join([form(x) for x in run])
                f.write(f"{writeStr}\n")

            if edit_number < (seq_edits - 1):
                edit_number += 1
            else:
                edit_number = 0
                model_number += 1
                model_edited.load_state_dict(model.state_dict())

            n_edits +=1
            if n_edits >= (5 * seq_edits):
                break
        print(f"Saved results to {saveloc}")
    if copy_to:
        shutil.copyfile(saveloc, f"{copy_to}/{filename}")

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

    def summary(self, long=False):
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
        if long:
            display(stats_df.melt())
        else:
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
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--lama', action='store_true')
    parser.add_argument('--kilt', action='store_true')
    parser.add_argument('--split_params', action='store_true')
    parser.add_argument('--edit_steps', default=5, type=int)
    parser.add_argument('--seq_edits', default=1, type=int)
    parser.add_argument('--model', type=str, default='bart-base')
    parser.add_argument('--copy_to')
    args = parser.parse_args()

    loc = utils.sailPreprocess()

    if args.model_path: 
        name = args.model if args.lama else 'gpt2'
        print(f"Using model {name}")
        model, tokenizer = utils.loadTrainedModel(args.model_path, name=name, cache_dir=loc,
                                                  split_params=args.split_params)

    ds = 'test' if args.test_set else 'validation'

    if args.gen:
        dataloader = utils.retrieveEditDataloader(
            tokenizer,
            bs=1,
            data_loc=loc,
            dataset=ds,
            self_sample=args.gen
        )
        evalSelfSample(
            model,
            dataloader,
            args.model_path,
            int(args.edit_steps),
            seq_edits=args.seq_edits,
            loc=loc,
            testset=args.test_set,
            copy_to=args.copy_to,
            lama=False
            )
    elif args.lama:
        dataloader = MaskedLMDataloader(
            'lama',
            tokenizer,
            loc=loc,
            bs=1,
            pct=40,
            shuffle=True,
            mode='editable',
            inner_loop='template',
            n_edits=1
        )
        validation = dataloader.validation
        evalSelfSample(
            model,
            validation,
            args.model_path,
            int(args.edit_steps),
            seq_edits=args.seq_edits,
            loc=loc,
            testset=args.test_set,
            copy_to=args.copy_to,
            lama=True
            )
    elif args.kilt:
        dataloader = MaskedLMDataloader(
            'kilt',
            tokenizer,
            loc=loc,
            bs=1,
            mode='editable',
            n_edits=1
        )
        validation = dataloader.validation
        evalSelfSample(
            model,
            validation,
            args.model_path,
            int(args.edit_steps),
            seq_edits=args.seq_edits,
            loc=loc,
            testset=args.test_set,
            copy_to=args.copy_to,
            lama=True
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
