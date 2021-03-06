import os
import tempfile
import argparse
import glob
import time
import random
from datetime import datetime
from collections import defaultdict
import logging
import string
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import transformers
import higher
import wandb

from torch.utils.data import Subset

import utils
from config import *
from evaluate import performOneEdit
from masked_lm.data import MaskedLMDataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)

RAND = ''.join(random.choice(string.ascii_letters) for i in range(5))
val = np.random.uniform()

class BaseTrainer:
    def __init__(self, config, train, validation, model_path=None):

        #configs
        self.config = config
        self.model_name = self.config.model_name
        self.model_dir = (
            f'{self.config.write_loc}/models/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
            )
        self.model, self.tokenizer = (
            utils.loadOTSModel(name=self.model_name, cache_dir=self.config.write_loc) if not model_path else
            utils.loadTrainedModel(model_path, name=self.model_name, cache_dir=self.config.write_loc)
            )

        #outfiles
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}"
        ewc = '_ewc' if self.config.ewc else ''
        split = '_split' if self.config.split_params else ''

        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}{RAND}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}{RAND}"
        self.statepath = (
            lambda model, epoch, step:
            f"{self.model_dir}/{model}{ewc}{split}_epoch{epoch}_ts{step}.{self.timestamp}{RAND}"
        )

        self.data = train
        self.validation_set = validation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.config.debug:
            wandb_dir = tempfile.mkdtemp()
            LOG.info(f"Writing wandb local logs to {wandb_dir}")
            wandb.init(
                project='patchable' if self.config.task == 'gen' else 'patchable_masked',
                entity='patchable-lm',
                config=self.config,
                name=f"{self.model_name}{ewc}{split}_{self.config.task}_{self.config.ds}_{self.timestamp}{RAND}",
                dir=wandb_dir,
                notes=self.config.notes,
            )
            transformers.logging.set_verbosity_info()

        self.epoch = 0

    def saveState(self, state_obj, train_step, name="finetune", final=False):
        if not self.config.debug:
            out_obj = state_obj.state_dict() if hasattr(state_obj, "state_dict") else state_obj
            if final:
                torch.save(
                    out_obj,
                    self.statepath(name, self.epoch, self.config.max_iter)
                    )
            elif (train_step > 0) and (train_step % self.config.model_save_pt == 0):
                torch.save(
                    out_obj,
                    self.statepath(name, self.epoch, train_step)
                    )

    def echo(self, train_step, **kwargs):
        if not self.config.silent:
            LOG.info((
                    f"Epoch: {self.epoch}; TrainStep {train_step}; ",
                    f"; ".join([f"{key} {val:0.4f}" for key,val in kwargs.items()])
                ))

    def tensorBoard(self, step, **kwargs):
        if not self.config.debug:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, step)

    def wandb_log(self, step, row):
        row['step'] = step
        if not self.config.debug:
            wandb.log(row)

    def mask_padding(self, tokens, mask, label=None):
        if self.config.task == 'gen':
            return tokens.to(self.device), mask.to(self.device), tokens.masked_fill(mask == 0, -100).to(self.device)
        elif self.config.task == 'cloze':
            return (tokens.to(self.device), mask.to(self.device),
            label.masked_fill(label == self.tokenizer.pad_token_id, -100).to(self.device))

    def strip_padding(self, tokens, mask, labels):
        mask_ = tokens != self.tokenizer.pad_token_id
        tokens = tokens[mask_].unsqueeze(0)
        mask = mask[mask_].unsqueeze(0)
        if mask_.shape == labels.shape:
            labels = labels[mask_].unsqueeze(0)
        else:
            label_mask = labels != self.tokenizer.pad_token_id
            labels = labels[label_mask].unsqueeze(0)

        return tokens.to(self.device), mask.to(self.device), labels.to(self.device)

    def get_edit_locs(self, tokens, labels):
        n_targets = (labels != -100).sum()
        edit_locs = torch.tensor([tokens.shape[-1] - n_targets - 1 + i for i in range(n_targets)]).to(self.device)
        gold_tokens = tokens[:,edit_locs]

        return edit_locs, gold_tokens.cpu()

    def validateBaseTraining(self):
        iters = 0
        loss = 0.0
        accuracy = 0.0

        with torch.no_grad():
            for train_step, datapack in enumerate(self.validation_set):
                if train_step >= 20:
                    break

                self.model.eval()
                if self.config.task == 'cloze':
                    (
                        lm_tokens, lm_mask, lm_labels,
                        _, _, _,
                        _, _, _,
                        _, _, _,
                    ) = datapack

                else:
                    lm_tokens, lm_mask = datapack
                    lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                    lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                
                lm_tokens_, lm_mask_, lm_labels_ = self.strip_padding(
                    lm_tokens[0], lm_mask[0], lm_labels[0]
                )

                base_out = self.model(
                    lm_tokens_,
                    attention_mask=lm_mask_,
                    labels=lm_labels_
                )

                loss += base_out.loss
                output_lps = F.log_softmax(base_out.logits, dim=-1)
                accuracy += (output_lps.argmax(-1) == lm_labels_).all(-1).float().mean().cpu()
                iters += 1

        loss /= iters
        accuracy /= iters
        self.echo(self.global_iter, **{'loss/val': loss, 'accuracy/val': accuracy})
        self.wandb_log(self.global_iter, {'loss/val': loss, 'accuracy/val': accuracy})

        self.model.train()

    def run(self):

        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(),
            self.config.outer_lr
        )

        self.global_iter = 0
        print("Starting Fine-tuning")

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            for train_step, datapack in enumerate(self.data):
                if self.config.task == 'cloze':
                    (
                        lm_tokens, lm_mask, lm_labels,
                        _, _, _,
                        _, _, _,
                        _, _, _,
                    ) = datapack
                    lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask, lm_labels)

                else:
                    lm_tokens, lm_mask = datapack
                    lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                    lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

                base_out = self.model(
                    lm_tokens,
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                l_base = base_out.loss
                l_base.backward()

                if train_step % self.config.bs == 0:
                    opt.step()
                    opt.zero_grad()

                if (train_step % self.config.val_interval == 0):
                    self.validateBaseTraining()

                self.global_iter += 1
                self.echo(train_step, **{"loss/base": l_base})
                self.wandb_log(self.global_iter, {"loss/base": l_base})
                self.saveState(self.model, train_step, self.model_name)


        self.saveState(self.model, train_step)



class EditTrainer(BaseTrainer):
    def __init__(self, config, train, validation, model_path=None):
        super().__init__(config, train, validation, model_path=None)
        self.val_iter = 0

        self.model = utils.loadTrainedModel(
            f"{self.config.write_loc}/models/finetune/{self.config.ft_model_name}",
            name=self.model_name,
            cache_dir=self.config.write_loc,
            tokenizer=False
        )

        self.model.eval()
        self.model.to(self.device)
    def validateEditTraining(self):
        self.model.eval()
        iters = 0

        for train_step, lm_data in enumerate(self.validation_set):
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

            base_out = self.model(
                lm_tokens,
                attention_mask=lm_mask,
                labels=lm_labels
            )
            self.writer.add_scalar("val_lbase", base_out.loss, self.val_iter)
            self.val_iter +=1
            iters += 1

            if iters >= 1000:
                break

        self.model.train()

    def perplexity(self, model, dataset):
        total_loss = []
        model.to(DEVICE)
        model.eval()
        acc = utils.NLLAccumulator()
        with torch.no_grad():
            for batch_idx, (lm_data, _, _, _) in enumerate(dataset):
                lm_tokens, lm_mask = lm_data[0]
                lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                out = model(lm_tokens, labels=lm_labels)

                loss = out.loss
                acc.update(loss.item(), acc.n_predictions_for_labels(lm_labels))
                total_loss.append(loss)

        avg_nll, ppl = acc.get_metrics()
        return torch.tensor(ppl)

    def run(self):

        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(),
            self.config.outer_lr
            )

        global_iter = 0
        LOG.info("Starting Training")

        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr))
            for p in self.model.transformer.h[-3:].parameters()
            ]
        lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            for train_step, (lm_data, edit_example, ent, old_ent) in enumerate(self.data):

                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

                edit_tokens, edit_mask = edit_example

                ent_tokens = ent[0].flatten()
                ent_tokens = ent_tokens[ent_tokens != 50256]
                edit_locs = utils.locateSubset(edit_tokens, ent_tokens)
                if edit_locs.nelement() == 0:
                    LOG.info(f"Unable to locate edit on TS {train_step}")
                    if not os.path.exists(f"{self.errpath}"):
                        os.mkdir(f"{self.errpath}")
                    torch.save(edit_tokens, f"{self.errpath}/edit_tokens_{train_step}")
                    torch.save(ent_tokens, f"{self.errpath}/ent_tokens_{train_step}")
                    continue

                edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
                edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
                edit_labels = edit_labels.to(self.device)
                edit_tokens, edit_mask = edit_tokens.to(self.device), edit_mask.to(self.device)

                param_groups = [
                    {'params': p, 'lr': None}
                    for p in self.model.transformer.h[-3:].parameters()
                    ]
                inner_opt = (
                    torch.optim.SGD(param_groups) if self.config.learnable_lr
                    else torch.optim.SGD(
                            self.model.transformer.h[-3:].parameters(),
                            lr=self.config.inner_lr
                        )
                    )

                with higher.innerloop_ctx(
                    self.model,
                    inner_opt,
                    override={'lr': self.lrs} if self.config.learnable_lr else None,
                    copy_initial_weights=False,
                    track_higher_grads=True
                    ) as (fmodel, diffopt):

                    for edit_step in range(self.config.n_edit_steps):

                        loss = fmodel(
                            edit_tokens,
                            attention_mask=edit_mask,
                            labels=edit_labels
                        ).loss
                        diffopt.step(loss)

                    edit_out = fmodel(
                        edit_tokens,
                        attention_mask=edit_mask,
                        labels=edit_labels
                    )
                    l_edit = edit_out.loss

                    base_out = self.model(
                        lm_tokens,
                        attention_mask=lm_mask,
                        labels=lm_labels
                    )
                    l_base = base_out.loss

                    edited_base_out = fmodel(
                        lm_tokens,
                        attention_mask=lm_mask,
                        labels=lm_labels
                    )

                    l_loc =  (
                        F.softmax(base_out.logits.detach(), dim=-1) *
                        (
                            F.log_softmax(base_out.logits.detach(), dim=-1) -
                            F.log_softmax(edited_base_out.logits, dim=-1)
                        )).sum(-1).mean()

                    total_loss = (
                        l_base +
                        self.config.cloc * l_loc  +
                        self.config.cedit * l_edit
                        )
                    total_loss.backward()

                # accumulate grads
                if train_step % self.config.bs == 0:
                    opt.step()
                    opt.zero_grad()

                    if self.config.learnable_lr:
                        lr_opt.step()
                        lr_opt.zero_grad()

                global_iter += 1

                loss_dict = {
                    "loss/base": l_base, "loss/edit": l_edit,
                    "loss/loc": l_loc, "loss/total": total_loss
                    }
                self.echo(train_step, **loss_dict)
                self.wandb_log(global_iter, loss_dict)
                self.saveState(self.model, global_iter, name='editable')
                if self.config.learnable_lr:
                    self.saveState(lr_opt, global_iter, name='lr')
                if global_iter >= self.config.max_iter:
                    LOG.info("Reached max iterations")
                    break

            # if (train_step % 1000 == 0) & (not self.config.debug):
            #     self.validateEditTraining()

        self.saveState(self.model, global_iter, final=True, name='editable')
        if self.config.learnable_lr:
            self.saveState(lr_opt, global_iter, final=True, name='lr')


class SelfSampleTrainer(EditTrainer):
    def __init__(self, config, train, validation, model_path=None):
        super().__init__(config, train, validation, model_path)

        if self.config.split_params:
            utils.wrap_model(self.model, self.model_name, self.config.ortho)


    def genModelText(self, lm_tokens):

        len_lm = lm_tokens.shape[-1]
        edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
        input_ids = lm_tokens[:, :edit_loc]
        input_size = input_ids.size()[-1]

        LOG.info("generating")
        output_sequence = self.model.generate(
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
        edit_labels = edit_labels.to(self.device)
        edit_tokens, edit_mask = edit_tokens.to(self.device), edit_mask.to(self.device)

        return edit_tokens, edit_mask, edit_labels

    def perplexity(self, model, dataset):
        total_loss = []
        model.to(DEVICE)
        model.eval()
        acc = utils.NLLAccumulator()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataset):
                lm_tokens, lm_mask, lm_labels = self.mask_padding(
                    batch[0], batch[1], batch[2] if self.config.task == 'cloze' else None
                    )
                loss = model(lm_tokens, labels=lm_labels).loss
                acc.update(loss.item(), acc.n_predictions_for_labels(lm_labels))

        avg_nll, ppl = acc.get_metrics()
        return torch.tensor(ppl)

    def validateSelfSampleTraining(self):
        self.model.eval()

        for ds in ['train', 'val']:

            data = self.validation_set if ds == 'val' else self.data

            ppl_pre_hist = []
            ppl_post_hist = []
            ll_change_hist = []
            accuracy_hist = []
            loss_hist = []

            indices = np.random.default_rng(self.val_iter).choice(len(data), 10, replace=False)
            subset = Subset(data, indices)
            for batch_idx, datapack in enumerate(subset):
                if self.config.task == 'gen':
                    (_, _, _, _, _, _, edit_tokens, edit_mask, edit_labels) = datapack

                elif self.config.task == 'cloze':
                    (
                        _, _, _,
                        _, _, _,
                        edit_tokens, edit_mask, edit_labels,
                        edit_template, edit_temp_mask, edit_labels
                    ) = datapack
                    edit_template = edit_template[:1]
                    edit_temp_mask = edit_temp_mask[:1]
                    edit_template, edit_temp_mask, _ = self.strip_padding(edit_template, edit_temp_mask, edit_labels)

                edit_tokens = edit_tokens[:1]
                edit_mask = edit_mask[:1]
                edit_labels = edit_labels[:1]

                edit_tokens, edit_mask, edit_labels = self.strip_padding(edit_tokens, edit_mask, edit_labels)

                if self.config.task == 'gen':
                    edit_locs, gold_tokens = self.get_edit_locs(edit_tokens, edit_labels)
                    edit_package = (edit_tokens, edit_mask, edit_labels)
                elif self.config.task == 'cloze':
                    edit_locs, gold_tokens = 0, edit_labels.flatten().unsqueeze(0)
                    edit_package = (edit_tokens, edit_mask, edit_labels, edit_template, edit_temp_mask, edit_labels)

                start = time.time()

                with torch.no_grad():
                    orig_ppl = self.perplexity(self.model, subset)

                model_out, logit_hist, ll_change, loss = performOneEdit(
                    self.model,
                    self.config.task,
                    self.lrs,
                    edit_package,
                    edit_locs,
                    gold_tokens[0],
                    n_edit_steps=1
                )
                with torch.no_grad():
                    new_ppl = self.perplexity(model_out, subset)

                ppl_pre_hist.append(orig_ppl.cpu())
                ppl_post_hist.append(new_ppl.cpu())
                ll_change_hist.append(ll_change)
                accuracy_hist.append(logit_hist[-1][-1])
                loss_hist.append(loss.detach().cpu())

            metrics = {
                f'ppl_pre/{ds}': np.mean(ppl_pre_hist),
                f'ppl_post/{ds}': np.mean(ppl_post_hist),
                f'll_change/{ds}': np.mean(ll_change_hist),
                f'accuracy/{ds}': np.mean(accuracy_hist),
                f'eval_loss/{ds}': np.mean(loss_hist),
            }
            if not self.config.debug:
                self.wandb_log(self.val_iter * self.config.val_interval, metrics)

        self.val_iter += 1

        self.model.train()

    def run(self):

        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.model.train()
        self.model.to(self.device)
        pg = [{'params': p, 'lr': self.config.outer_lr if hasattr(p, '__conditioner__') else self.config.outer_lr} for p in self.model.parameters()]
        opt = torch.optim.Adam(pg)

        global_iter = 0
        LOG.info("Starting Training")

        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr))
            for p in self.model.inner_params()
            ]
        lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)

        skip_count = 0
        info_dict_ = defaultdict(list)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            for train_step, datapack in enumerate(self.data):
                if self.config.task == 'gen':
                    (lm_tokens, lm_mask, loc_tokens, loc_mask, _, _, edit_inner, edit_inner_mask, edit_labels) = datapack
                    lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask)
                    loc_tokens, loc_mask, loc_labels = self.mask_padding(loc_tokens, loc_mask)
                    edit_outer, edit_outer_mask = edit_inner, edit_inner_mask
                elif self.config.task == 'cloze':
                    (
                        lm_tokens, lm_mask, lm_labels,
                        loc_tokens, loc_mask, loc_labels,
                        edit_outer, edit_outer_mask, edit_labels,
                        edit_inner, edit_inner_mask, edit_labels,
                    ) = datapack

                    lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask, lm_labels)
                    loc_tokens, loc_mask, loc_labels = self.mask_padding(loc_tokens, loc_mask, loc_labels)

                # Cache the current params and grads since we're going to modify the model during
                #  the edit process
                p_cache = {}
                for n, p in self.model.named_parameters():
                    p_cache[n] = p.data.detach().clone()

                for edit_example_idx in range(self.config.n_edits):
                    if self.config.noise_coef is not None:
                        for p in self.model.inner_params():
                            p.data += torch.randn_like(p.data) * self.config.noise_coef
                        
                    edit_inner_tokens_, edit_inner_mask_, edit_labels_ = (
                        self.strip_padding(edit_inner[edit_example_idx],
                                           edit_inner_mask[edit_example_idx],
                                           edit_labels[edit_example_idx])
                    )

                    edit_outer_tokens_, edit_outer_mask_, _ = (
                        self.strip_padding(edit_outer[edit_example_idx],
                                           edit_outer_mask[edit_example_idx],
                                           edit_labels[edit_example_idx])
                    )

                    param_groups = [
                        {'params': p, 'lr': None}
                        for p in self.model.inner_params()
                    ]
                    inner_opt = (
                        torch.optim.SGD(param_groups) if self.config.learnable_lr
                        else torch.optim.SGD(
                                self.model.inner_params(),
                                lr=self.config.inner_lr
                        )
                    )

                    with higher.innerloop_ctx(
                            self.model,
                            inner_opt,
                            override={'lr': self.lrs} if self.config.learnable_lr else None,
                            copy_initial_weights=False,
                            track_higher_grads=True
                    ) as (fmodel, diffopt):
                        for edit_step in range(self.config.n_edit_steps):
                            if self.config.split_params:
                                fmodel.set_editing(True)
                            loss = fmodel(
                                edit_inner_tokens_,
                                attention_mask=edit_inner_mask_,
                                labels=edit_labels_
                            ).loss
                            if self.config.split_params:
                                fmodel.set_editing(False)
                            diffopt.step(loss)

                        edit_out = fmodel(
                            edit_outer_tokens_,
                            attention_mask=edit_outer_mask_,
                            labels=edit_labels_
                        )
                        l_edit = edit_out.loss

                        edited_base_out = fmodel(
                            loc_tokens[edit_example_idx].unsqueeze(0),
                            attention_mask=loc_mask[edit_example_idx].unsqueeze(0),
                            labels=loc_labels[edit_example_idx].unsqueeze(0)
                        )

                        with torch.no_grad():
                            base_out = self.model(
                                loc_tokens[edit_example_idx].unsqueeze(0),
                                attention_mask=loc_mask[edit_example_idx].unsqueeze(0),
                                labels=loc_labels[edit_example_idx].unsqueeze(0)
                            )

                        l_loc = (
                            base_out.logits.softmax(-1).detach() * (
                                base_out.logits.log_softmax(-1).detach() -
                                edited_base_out.logits.log_softmax(-1)
                            )
                        ).sum(-1).mean()

                        total_edit_loss = (
                            self.config.cloc * l_loc  +
                            self.config.cedit * l_edit
                        ) / self.config.n_edits

                        if not self.config.split_params:
                            total_edit_loss.backward()
                        else:
                            # Only train phi/lrs using edit loss, not theta
                            edit_params = self.model.phi() + self.lrs
                            for p, g in zip(edit_params, A.grad(total_edit_loss, edit_params)):
                                if p.grad is not None:
                                    p.grad += g
                                else:
                                    p.grad = g.clone()

                        for fp, p in zip(fmodel.parameters(), self.model.parameters()):
                            p.data = fp.data.detach()

                    # It only makes sense to train more than one edit if we've split the params
                    if not self.config.split_params:
                        break

                # restore pre-edit parameters without overwriting the gradients we just computed
                for n, p in self.model.named_parameters():
                    p.data = p_cache[n]

                # Compute base loss
                base_out = self.model(
                    lm_tokens,
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                l_base = base_out.loss
                l_base.backward()

                global_iter += 1

                info_dict = {
                    "loss/base": l_base, "loss/edit": l_edit,
                    "loss/loc": l_loc, "loss/train": total_edit_loss + l_base,
                }

                if self.config.split_params:
                    info_dict["grad/phi"] = torch.nn.utils.clip_grad_norm_(self.model.phi(), 50)
                    info_dict["grad/theta"] = torch.nn.utils.clip_grad_norm_(self.model.theta(), 50)
                    info_dict['grad/lrs'] = torch.nn.utils.clip_grad_norm_(self.lrs, 50).item()
                else:
                    info_dict["grad/all"] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                    info_dict['grad/lrs'] = torch.nn.utils.clip_grad_norm_(self.lrs, 50).item()

                for k, v in info_dict.items():
                    info_dict_[k].append(v)
                
                # accumulate grads
                if train_step % 5 == 0:
                    means = {k: sum(v) / len(v) for (k,v) in info_dict_.items()}
                    self.echo(train_step, **means)
                    means.update({f"lr/lr{i}":lr.data.item() for i, lr in enumerate(self.lrs)})
                    self.wandb_log(global_iter, means)
                    info_dict_ = defaultdict(list)
                    
                    opt.step()
                    opt.zero_grad()

                    if self.config.learnable_lr:
                        lr_opt.step()
                        lr_opt.zero_grad()

                if (train_step % self.config.val_interval == 0):
                    self.validateSelfSampleTraining()
                
                self.saveState(self.model, global_iter, name=f"{self.model_name}_{self.config.task}_{self.config.ds}")
                if self.config.learnable_lr:
                    self.saveState(self.lrs, global_iter, name=f'lr_{self.model_name}')
                if global_iter >= self.config.max_iter:
                    LOG.info("Reached max iterations")
                    break

        self.saveState(self.model, global_iter, final=True, name=f"{self.model_name}_{self.config.task}_{self.config.ds}")
        if self.config.learnable_lr:
            self.saveState(self.lrs, global_iter, final=True, name=f'lr_{self.model_name}')


class EWCTrainer(SelfSampleTrainer):
    def __init__(self, config, train, validation, weight=1e-7, model_path=None):
        super().__init__(config, train, validation, model_path=None)

        self.weight = weight
        self._update_mean()
        self._update_fisher(self.data)

    def _update_mean(self):
        LOG.info("Estimating parameter means")
        for n, p in self.model.named_parameters():
            buffer_name = n.replace('.', '__')
            self.model.register_buffer(buffer_name + '_mean', p.data.clone())

    def _update_fisher(self, dataloader, n_batches=10000):
        LOG.info("Estimating parameter Fisher matrices")
        sq_grad_sums = [None] * len(list(self.model.parameters()))  # Store the sum of squared gradients of log likelihoods
        n_samples = 0
        for batch_idx, datapack in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            if self.config.task == 'gen':
                (lm_tokens, lm_mask, _, _, _, _, _, _, _) = datapack
                lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask)

            elif self.config.task == 'cloze':
                (
                    lm_tokens, lm_mask, lm_labels,
                    _, _, _,
                    _, _, _,
                    _, _, _,
                ) = datapack

            lm_tokens_, lm_mask_, lm_labels_ = self.strip_padding(lm_tokens, lm_mask, lm_labels)

            model_out = self.model(
                lm_tokens_,
                attention_mask=lm_mask_,
                labels=lm_labels_
            )
            lps = F.log_softmax(model_out.logits, -1).gather(-1, lm_labels_.view(1, 1, -1))
            ll = torch.sum(lps)
            for g_idx, g in enumerate(A.grad(ll, self.model.parameters())):
                sq_grad_sums[g_idx] = g.data.clone()**2 if sq_grad_sums[g_idx] is None else (sq_grad_sums[g_idx] + g.data.clone()**2)
            n_samples += lm_tokens.shape[0]

        for (n, p), sq_grad_sum in zip(self.model.named_parameters(), sq_grad_sums):
            buffer_name = n.replace('.', '__')
            sq_grad = sq_grad_sum / n_samples
            self.model.register_buffer(buffer_name + '_fisher', sq_grad)


    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.model.train()
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), self.config.outer_lr)

        global_iter = 0
        LOG.info("Starting Training")

        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr))
            for p in self.model.inner_params()
            ]
        lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)

        skip_count = 0
        info_dict_ = defaultdict(list)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            for train_step, datapack in enumerate(self.data):
                if self.config.task == 'gen':
                    (lm_tokens, lm_mask, loc_tokens, loc_mask, _, _, edit_inner, edit_inner_mask, edit_labels) = datapack
                    lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask)
                    loc_tokens, loc_mask, loc_labels = self.mask_padding(loc_tokens, loc_mask)
                    edit_outer, edit_outer_mask = edit_inner, edit_inner_mask
                elif self.config.task == 'cloze':
                    (
                        lm_tokens, lm_mask, lm_labels,
                        loc_tokens, loc_mask, loc_labels,
                        edit_outer, edit_outer_mask, edit_labels,
                        edit_inner, edit_inner_mask, edit_labels,
                    ) = datapack

                    lm_tokens, lm_mask, lm_labels = self.mask_padding(lm_tokens, lm_mask, lm_labels)
                    loc_tokens, loc_mask, loc_labels = self.mask_padding(loc_tokens, loc_mask, loc_labels)

                for edit_example_idx in range(self.config.n_edits):
                    edit_inner_tokens_, edit_inner_mask_, edit_labels_ = (
                        self.strip_padding(edit_inner[edit_example_idx],
                                            edit_inner_mask[edit_example_idx],
                                            edit_labels[edit_example_idx])
                    )

                    edit_outer_tokens_, edit_outer_mask_, _ = (
                        self.strip_padding(edit_outer[edit_example_idx],
                                            edit_outer_mask[edit_example_idx],
                                            edit_labels[edit_example_idx])
                    )
                    
                    param_groups = [
                        {'params': p, 'lr': None}
                        for p in self.model.inner_params()
                    ]
                    inner_opt = (
                        torch.optim.SGD(param_groups) if self.config.learnable_lr
                        else torch.optim.SGD(
                                self.model.inner_params(),
                                lr=self.config.inner_lr
                        )
                    )

                    with higher.innerloop_ctx(
                            self.model,
                            inner_opt,
                            override={'lr': self.lrs} if self.config.learnable_lr else None,
                            copy_initial_weights=False,
                            track_higher_grads=True
                    ) as (fmodel, diffopt):
                        for edit_step in range(self.config.n_edit_steps):
                            if self.config.split_params:
                                fmodel.set_editing(True)
                            loss = fmodel(
                                edit_inner_tokens_,
                                attention_mask=edit_inner_mask_,
                                labels=edit_labels_
                            ).loss
                            if self.config.split_params:
                                fmodel.set_editing(False)
                            diffopt.step(loss)

                        edit_out = fmodel(
                            edit_outer_tokens_,
                            attention_mask=edit_outer_mask_,
                            labels=edit_labels_
                        )
                        l_edit = edit_out.loss

                        edited_base_out = fmodel(
                            loc_tokens[edit_example_idx].unsqueeze(0),
                            attention_mask=loc_mask[edit_example_idx].unsqueeze(0),
                            labels=loc_labels[edit_example_idx].unsqueeze(0)
                        )

                        with torch.no_grad():
                            base_out = self.model(
                                loc_tokens[edit_example_idx].unsqueeze(0),
                                attention_mask=loc_mask[edit_example_idx].unsqueeze(0),
                                labels=loc_labels[edit_example_idx].unsqueeze(0)
                            )

                        l_loc = (
                            base_out.logits.softmax(-1).detach() * (
                                base_out.logits.log_softmax(-1).detach() -
                                edited_base_out.logits.log_softmax(-1)
                            )
                        ).sum(-1).mean()

                        total_edit_loss = (
                            self.config.cloc * l_loc  +
                            self.config.cedit * l_edit
                        ) / self.config.n_edits
                        total_edit_loss.backward()

                    # Only perform one edit
                    break

                # Compute quadratic loss
                l_base = 0.0
                for n, p in self.model.named_parameters():
                    buffer_name = n.replace('.', '__')
                    mean = getattr(self.model, f'{buffer_name}_mean')
                    fisher = getattr(self.model, f'{buffer_name}_fisher')
                    l_base += self.weight * 0.5 * (fisher * (p - mean) ** 2).sum()
                l_base.backward()

                global_iter += 1

                info_dict = {
                    "loss/base": l_base, "loss/edit": l_edit,
                    "loss/loc": l_loc, "loss/train": total_edit_loss + l_base,
                }

                info_dict["grad/all"] = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                info_dict['grad/lrs'] = torch.nn.utils.clip_grad_norm_(self.lrs, 50).item()

                for k, v in info_dict.items():
                    info_dict_[k].append(v)
                
                # accumulate grads
                if train_step % 5 == 0:
                    means = {k: sum(v) / len(v) for (k,v) in info_dict_.items()}
                    self.echo(train_step, **means)
                    means.update({f"lr/lr{i}":lr.data.item() for i, lr in enumerate(self.lrs)})
                    self.wandb_log(global_iter, means)
                    info_dict_ = defaultdict(list)
                    
                    opt.step()
                    opt.zero_grad()

                    if self.config.learnable_lr:
                        lr_opt.step()
                        lr_opt.zero_grad()

                if (train_step % self.config.val_interval == 0):
                    self.validateSelfSampleTraining()
                
                self.saveState(self.model, global_iter, name=f"{self.model_name}_{self.config.task}_{self.config.ds}")
                if self.config.learnable_lr:
                    self.saveState(self.lrs, global_iter, name=f'lr_{self.model_name}')
                if global_iter >= self.config.max_iter:
                    LOG.info("Reached max iterations")
                    break

        self.saveState(self.model, global_iter, final=True, name=f"{self.model_name}_{self.config.task}_{self.config.ds}")
        if self.config.learnable_lr:
            self.saveState(self.lrs, global_iter, final=True, name=f'lr_{self.model_name}')


if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--n_edits', type=int, default=1)
    parser.add_argument('--split_params', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--ewc', action='store_true')
    parser.add_argument('--ewc_weight', type=float, default=1e-7)
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--lama', action='store_true')
    parser.add_argument('--kilt', action='store_true')
    parser.add_argument('--ortho', action='store_true', default=False)
    parser.add_argument('--bs', default=5, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--val_interval', type=int, default=200)
    parser.add_argument('--outer_lr', type=float, default=None)
    parser.add_argument('--inner_lr', type=float, default=None)
    parser.add_argument('--cedit', type=float, default=None)
    parser.add_argument('--cloc', type=float, default=None)
    parser.add_argument('--model', type=str, default='bart-base')
    parser.add_argument('--noise_coef', type=float, default=None)
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--lr_lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    loc = utils.sailPreprocess()
    tokenizer = utils.loadTokenizer(
        name= 'gpt2' if args.gen else args.model,
        cache_dir=loc)

    config = (
        SelfSampleGPT2Config() if args.gen else
        LamaBartConfig() if args.model == 'bart-base' and args.lama else
        KiltBartConfig() if args.model == 'bart-base' and args.kilt else
        TrainConfig() if args.finetune else
        ClozeT5Config()
    )
    config.write_loc = loc
    config.bs = args.bs
    config.n_edits = args.n_edits
    config.ewc = args.ewc
    config.ewc_weight = args.ewc_weight
    config.split_params = args.split_params
    config.debug = args.debug if args.debug else config.debug
    config.val_interval = args.val_interval
    config.ds = 'kilt' if args.kilt else 'lama' if args.lama else 'wikitext'
    config.ortho = args.ortho
    config.noise_coef = args.noise_coef
    config.notes = args.notes
    if args.outer_lr is not None:
        LOG.info(f"Overriding default outer_lr {config.outer_lr} with new value {args.outer_lr}")
        config.outer_lr = args.outer_lr
    if args.inner_lr is not None:
        LOG.info(f"Overriding default inner_lr {config.inner_lr} with new value {args.inner_lr}")
        config.inner_lr = args.inner_lr
    if args.lr_lr is not None:
        LOG.info(f"Overriding default lr_lr {config.lr_lr} with new value {args.lr_lr}")
        config.lr_lr = args.lr_lr
    if args.cedit is not None:
        LOG.info(f"Overriding default cedit {config.cedit} with new value {args.cedit}")
        config.cedit = args.cedit
    if args.cloc is not None:
        LOG.info(f"Overriding default cloc {config.cloc} with new value {args.cloc}")
        config.cloc = args.cloc
    if args.epochs is not None:
        LOG.info(f"Overriding default epochs {config.epochs} with new value {args.epochs}")
        config.epochs = args.epochs

    if args.gen:
        train = utils.retrieveUnifiedDataset(
            tokenizer,
            data_loc=config.write_loc,
            bs=1,
            dataset='train',
            self_sample=True,
            n_edits=args.n_edits
        )
        validation = utils.retrieveUnifiedDataset(
            tokenizer,
            data_loc=config.write_loc,
            bs=1,
            dataset='validation',
            self_sample=True,
            n_edits=args.n_edits
        )
    elif args.lama:
        dataloader = MaskedLMDataloader(
            'lama',
            tokenizer,
            loc=loc,
            bs=args.bs,
            pct=40,
            shuffle=True,
            mode='editable',
            inner_loop=getattr(config, 'inner_loop', 'template'),
            n_edits=args.n_edits
        )
        train = dataloader.train
        validation = dataloader.validation
    elif args.kilt:
        dataloader = MaskedLMDataloader(
            'kilt',
            tokenizer,
            loc=loc,
            bs=args.bs,
            mode='editable',
            n_edits=args.n_edits
        )
        train = dataloader.train
        validation = dataloader.validation

    else:
        dataloader = utils.wikiDataloader(
            tokenizer,
            bs=args.bs,
            data_loc=loc,
            dataset='train',
            shuffle=False,
            max_length=200,
            min_length=20
        )

    if args.finetune:
        trainer = BaseTrainer(config, train, validation)

    elif args.ewc:
        trainer = EWCTrainer(config, train, validation, args.ewc_weight, tokenizer)

    elif (args.gen or args.lama or args.kilt):
        trainer = SelfSampleTrainer(config, train, validation, tokenizer)

    else:
        raise AttributeError("Must specify train arg")

    trainer.run()
