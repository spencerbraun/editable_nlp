import random
import copy
import argparse
import os
import re
import shutil
from datetime import datetime
import logging

import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import transformers
from torch.utils.data import Subset

import utils
from data_process import TorchDataset
from masked_lm.data import MaskedLMDataloader

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOKENIZER = None

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def get_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def processClozeBatch(tokens, mask, label, pad_token_id=0):
    return (tokens.to(DEVICE), mask.to(DEVICE),
            label.masked_fill(label == pad_token_id, -100).to(DEVICE))

def strip_padding(tokens, mask, labels, pad_token_id=0):
    mask_ = tokens != pad_token_id
    tokens = tokens[mask_].unsqueeze(0)
    mask = mask[mask_].unsqueeze(0)

    label_mask = labels != pad_token_id
    labels = labels[label_mask].unsqueeze(0)

    return tokens.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)


def perplexity(model, dataloader, cloze=False, iteration=0, pad_token_id=0):
    model.to(DEVICE)
    model.eval()
    acc = utils.NLLAccumulator()

    indices = np.random.default_rng(iteration).choice(len(dataloader), 200, replace=False)
    subset = Subset(dataloader, indices)
    dataset = dataloader if not cloze else subset
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataset):
            if cloze:
                lm_tokens, lm_mask, lm_labels = processClozeBatch(
                    batch[0], batch[1], batch[2], pad_token_id=pad_token_id
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

def cloze_performance(model, dataloader, iteration=0, pad_token_id=0):
    model.to(DEVICE)
    model.eval()
    acc = utils.NLLAccumulator()

    indices = np.random.default_rng(iteration).choice(len(dataloader), 200, replace=False)
    subset = Subset(dataloader, indices)

    logit_sum = 0
    accs = []
    total_n = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(subset):
            (
            lm_tokens, lm_mask, lm_labels,
            _, _, _,
            _, _, _,
            _, _, _,
            ) = batch
            lm_tokens_, lm_mask_, lm_labels_ = strip_padding(lm_tokens, lm_mask, lm_labels, pad_token_id)
            logit, accuracy = getClozeIndexedProbs(model, 0, lm_labels_,lm_tokens_, lm_labels_, silent=False)
            accs.append(accuracy)
            logit_sum += logit
            total_n += acc.n_predictions_for_labels(lm_labels_, offset=0)

    return np.mean(accs), logit_sum / total_n

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

def getClozeIndexedProbs(model, index, gold_tokens, sent_tokens, labels, silent=False):

    model.eval()
    with torch.no_grad():
        output = model(input_ids=sent_tokens, labels=labels)
        output_lps = F.log_softmax(output.logits, dim=-1)
        logits = output_lps.detach().cpu().squeeze(0) #squeeze batch size

        gold_tokens = gold_tokens.flatten().unsqueeze_(1).cpu()
        logit_sum = torch.sum(logits.gather(1, gold_tokens))
        accuracy = (output_lps.argmax(-1) == labels).all(-1).float().mean().cpu()

        if not silent:
            global TOKENIZER
            if TOKENIZER is None:
                TOKENIZER = utils.loadTokenizer(name='bart-base')
            LOG.info('*'*50)
            LOG.info(f"GOLD: {TOKENIZER.batch_decode(labels)}")
            LOG.info(f"GUESS: {TOKENIZER.batch_decode(output_lps.argmax(-1))}")
            LOG.info('*'*50)

    model.train()
    return (logit_sum, accuracy)

def loadLr(model_path, lr_path=None):

    if not lr_path:
        dir_loc = os.path.dirname(model_path)

        if '_split' in model_path:
            idx = model_path.rindex('split')
        else:
            idx = model_path.rindex('epoch')
        ext = model_path[idx:]

        lr_glob = glob.glob(f"{dir_loc}/lr_*{ext}")
        
        if len(lr_glob) > 1:
            raise AttributeError("Too many lr specifications", ",".join(lr_glob))
        elif len(lr_glob) == 0:
            raise AttributeError("No lr specifications found")
        else:
            LOG.info(f"Loading lrs {lr_glob[0]}")
            lrs = torch.load(lr_glob[0])
    else:
        LOG.info(f"Loading passed lrs {lr_path}")
        lrs = torch.load(lr_path)

    return lrs

def performOneEdit(
    model,
    task,
    lrs,
    edit_package,
    edit_locs,
    gold_tokens,
    n_edit_steps=10,
    mode="val",
    delta=None
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
    inner_opt = torch.optim.Adam(param_groups, lr=1e-5) if mode == "mmtm" else torch.optim.SGD(param_groups)

    if task == 'gen':
        edit_tokens, edit_mask, edit_labels = edit_package
        idxProbs = getIndexedProbs
    elif task == 'cloze':
        (
            edit_outer, edit_outer_mask, edit_labels,
            edit_inner, edit_inner_mask, edit_labels,
        ) = edit_package

        idxProbs = getClozeIndexedProbs

    logit_hist = []
    if task == 'gen':
        logit_hist.append(
            idxProbs(model, edit_locs, gold_tokens, edit_tokens, None)
        )
    elif task == 'cloze':
        logit_hist.append(
            idxProbs(model, edit_locs, gold_tokens, edit_outer, edit_labels)
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

            if task == 'gen':
                output = fmodel(
                    edit_tokens,
                    attention_mask=edit_mask,
                    labels=edit_labels
                )
            elif task == 'cloze':
                output = fmodel(
                    edit_inner,
                    attention_mask=edit_inner_mask,
                    labels=edit_labels
                )

            if hasattr(fmodel, "set_editing"):
                fmodel.set_editing(False)
            diffopt.step(output.loss)

            if mode == 'mmtm':
                for p, fp in zip(model.parameters(), fmodel.parameters()):
                    orig_p = p.data.detach()
                    edited_p = fp.data.detach()
                    fp.data = (
                        orig_p + torch.clamp(
                            torch.clamp(edited_p - orig_p, min=-delta),
                            max=delta
                        )
                    )

            if task == 'gen':
                logit_hist.append(
                    idxProbs(fmodel, edit_locs, gold_tokens, edit_tokens, None),
                )
            elif task == 'cloze':
                logit_hist.append(
                    idxProbs(fmodel, edit_locs, gold_tokens, edit_outer, edit_labels),
                )

        ll_change = (abs(logit_hist[0][0]) - abs(logit_hist[-1][0]))/abs(logit_hist[0][0])
        prob_change = logit_hist[-1][0].exp() - logit_hist[0][0].exp()
        LOG.info(f"prob history: {[l[0].exp() for l in logit_hist]}")
        LOG.info(f"Edit step {edit_step}; d_prob {prob_change}; ll change {ll_change}; logit {logit_hist[-1][0]}; loss {output.loss}")

    edited_model = copy.deepcopy(model)
    edited_model.load_state_dict(fmodel.state_dict())

    return edited_model, logit_hist, ll_change, output.loss

def genModelText(finetuned, lm_tokens):

    len_lm = lm_tokens.shape[-1]
    edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
    input_ids = lm_tokens[:, :edit_loc]
    input_size = input_ids.size()[-1]

    finetuned.eval()
    LOG.info(f"generating, {DEVICE}")
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
            LOG.info(f"TS {train_step}")

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


def processBatch(batch, cloze=False, pad_token_id=0):
    if cloze:
        (
            _, _, _,
            _, _, _,
            edit_tokens, edit_mask, edit_labels,
            edit_template, edit_temp_mask, edit_labels
        ) = batch

        edit_template = edit_template[:1]
        edit_temp_mask = edit_temp_mask[:1]

        edit_tokens, edit_mask, _ = strip_padding(edit_tokens, edit_mask, edit_labels, pad_token_id)
        edit_template, edit_temp_mask, edit_labels = strip_padding(edit_template, edit_temp_mask, edit_labels, pad_token_id)

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

def getPadTokenID(model):

    if isinstance(model, transformers.BartForConditionalGeneration):
        return 1
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        return 0
    elif isinstance(model, transformers.GPT2LMHeadModel):
        return 50256

def evalSelfSample(
    model,
    dataloader,
    model_name,
    n_edit_steps,
    seq_edits=1,
    loc="..",
    testset=False,
    copy_to="",
    cloze=False,
    mmtm=False,
    delta=None,
    lr_path=None,
    n_runs=5,
    stats_freq=20
    ):

    pad_token_id = getPadTokenID(model)
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    mmtm = f'_mmtm{delta}' if mmtm else ''
    filename = f"edit_success{mmtm}_{timestamp}_{os.path.basename(model_name)}"
    saveloc = f"{loc}/eval/{filename}" if not testset else f"{loc}/eval/test/{filename}"

    if model_name.startswith('OTS'):
        lrs = []  # Don't try to load lrs for OTS models
    else:
        try:
            lrs = loadLr(model_name, lr_path)
        except AttributeError as e:
            LOG.info(e)
            LOG.info("No learning rates found!")
            lrs = []

    n_edits = 0
    edit_number = 0
    model_number = 0
    sequential = seq_edits > 1

    model_edited = copy.deepcopy(model).to(DEVICE)

    orig_ppl = perplexity(model, dataloader, cloze=cloze, pad_token_id=pad_token_id)
    if cloze:
        orig_acc, orig_lp = cloze_performance(model, dataloader, iteration=n_edits, pad_token_id=pad_token_id)
    else:
        orig_acc, orig_lp = "", ""
    orig_params = get_params(model)

    with open(saveloc, "w") as f:
        f.write(
            "model_number,edit_number,train_step,n_edit_steps,edit_step,"
            "logits,orig_ppl,new_ppl,orig_acc,new_acc,orig_lp,new_lp,norm,edit_accuracy\n"
            )
        for train_step, batch in enumerate(dataloader):
            LOG.info(f"Val Step {train_step}")
            LOG.info(f"Edit number {edit_number}")
            LOG.info(f"Model number {model_number}")
            edit_package, edit_locs, gold_tokens = processBatch(batch, cloze=cloze, pad_token_id=pad_token_id)

            model_edited, logit_hist, ll_change, loss = performOneEdit(
                model_edited,
                'cloze' if cloze else 'gen',
                lrs,
                edit_package,
                edit_locs,
                gold_tokens,
                n_edit_steps=n_edit_steps,
                mode=("mmtm" if mmtm else "val"),
                delta=(delta if mmtm else None)
            )

            if ((edit_number+1) % stats_freq == 0) or (edit_number == 0):
                if cloze:
                    new_acc, new_lp = cloze_performance(model_edited, dataloader, iteration=n_edits, pad_token_id=pad_token_id)
                    new_ppl = ""
                else:
                    new_ppl = perplexity(model_edited, dataloader, cloze=cloze, iteration=n_edits, pad_token_id=pad_token_id)
                    new_acc, new_lp = "", ""
            else:
                new_ppl, new_acc, new_lp = "", "", ""
                

            norm_diff = orig_params.sub(get_params(model_edited)).norm().item()

            for idx, (val, acc) in enumerate(logit_hist):
                run = (
                    model_number,edit_number,train_step, n_edit_steps, idx, val,
                    orig_ppl,new_ppl,orig_acc,new_acc,orig_lp,new_lp,norm_diff,acc
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
            if n_edits >= (n_runs * seq_edits):
                break
        LOG.info(f"Saved results to {saveloc}")
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
        LOG.info("Model Parameters:")
        display(self.modelStats)

        LOG.info("Success Metrics")

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
        LOG.info("Plotting Logits")
        for name, model in self.models.items():
            LOG.info(name)

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
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='')
    parser.add_argument('--lr_path', default=None)
    parser.add_argument('--test_set', action='store_true')
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--lama', action='store_true')
    parser.add_argument('--kilt', action='store_true')
    parser.add_argument('--ortho', action='store_true')
    parser.add_argument('--split_params', action='store_true')
    parser.add_argument('--edit_steps', default=1, type=int)
    parser.add_argument('--seq_edits', default=1, type=int)
    parser.add_argument('--stats_freq', default=10, type=int)
    parser.add_argument('--model', type=str, default='bart-base')
    parser.add_argument('--mmtm', action='store_true')
    parser.add_argument('--delta', type=float, default=5.0e-4, help='Delta for MMTM')
    parser.add_argument('--copy_to')
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    loc = utils.sailPreprocess()
    name = args.model if not args.gen else 'gpt2'
    if args.model_path:
        LOG.info(f"Using model {name}")
        model, tokenizer = utils.loadTrainedModel(
            args.model_path,
            name=name,
            cache_dir=loc,
            split_params=args.split_params,
            ortho=args.ortho
        )
    else:
        LOG.info(f"Using OTS {name} model")
        model, tokenizer = utils.loadOTSModel(name=name, cache_dir=loc)
        utils.prep_for_maml(model)
        args.model_path = f'OTS_{name}'

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
            cloze=False,
            mmtm=args.mmtm,
            delta=args.delta,
            n_runs=args.n_runs,
            stats_freq=args.stats_freq
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
            cloze=True,
            lr_path=args.lr_path,
            mmtm=args.mmtm,
            delta=args.delta,
            n_runs=args.n_runs,
            stats_freq=args.stats_freq
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
            cloze=True,
            lr_path=args.lr_path,
            mmtm=args.mmtm,
            delta=args.delta,
            n_runs=args.n_runs,
            stats_freq=args.stats_freq
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
