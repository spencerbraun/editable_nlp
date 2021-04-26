import os
import argparse
import glob
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import higher
import wandb

from torch.utils.data import Subset

import utils
from config import TrainConfig, EditConfig, SelfSampleConfig
from evaluate import performOneEdit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseTrainer:
    def __init__(self, config, dataloader, model_path=None):

        #configs
        self.config = config
        self.model_dir = (
            f'{self.config.write_loc}/models/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
            )
        self.model, self.tokenizer = (
            utils.loadOTSModel(cache_dir=self.config.write_loc) if not model_path else 
            utils.loadTrainedModel(model_path, cache_dir=self.config.write_loc)
            )

        #outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}"
        self.statepath = (
            lambda model, epoch, step: 
            f"{self.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
            )

        self.data = dataloader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self.config.debug:
            wandb.init(
                project='patchable',
                entity='patchable-lm',
                config=self.config,
                name=f"{self.config.task}_{self.timestamp}",
                dir=self.config.write_loc,
            )
            wandb.watch(self.model)
            transformers.logging.set_verbosity_error()

        self.epoch = 0

    def saveState(self, state_obj, train_step, name="finetune", final=False):
        if not self.config.debug:
            out_obj = state_obj.state_dict() if hasattr(state_obj, "state_dict") else state_obj
            if final:
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, 9999)
                    )
            elif (train_step > 0) & (train_step % self.config.model_save_pt == 0):
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, train_step)
                    )

    def echo(self, train_step, **kwargs):
        if not self.config.silent:
            print((
                    f"Epoch: {self.epoch}; TrainStep {train_step}; ",
                    f"; ".join([f"{key} {val}" for key,val in kwargs.items()])
                )) 

    def tensorBoard(self, step, **kwargs):
        if not self.config.debug:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, step)
    
    def wandb_log(self, step, row):
        row['step'] = step
        if not self.config.debug:
            wandb.log(row)

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
        print("Starting Fine-tuning")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            for train_step, lm_data in enumerate(self.data):
                
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                
                base_out = self.model(
                    lm_tokens, 
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                l_base = base_out.loss
                l_base.backward()

                if train_step % 5 == 0:
                    opt.step()
                    opt.zero_grad()
                
                global_iter += 1
                self.echo(train_step, **{"loss/base": l_base})
                self.wandb_log(global_iter, {"loss/base": l_base})
                self.saveState(self.model, train_step, "gpt2")


        self.saveState(self.model, train_step)



class EditTrainer(BaseTrainer):
    def __init__(self, config, dataloader, model_path=None):
        super().__init__(config, dataloader, model_path=None) 
        self.validation_set = utils.wikiDataloader(
            self.tokenizer, 
            bs=15, 
            data_loc=self.config.write_loc,
            dataset='validation',
            shuffle=True
        )
        self.val_iter = 0

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

    def perplexity(self, model, data):
        total_loss = []
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            indices = [idx % len(data) for idx in range(self.val_iter, self.val_iter + 100)]  # select 100 elements
            subset = Subset(dataloader.dataset, indices)
            for batch_idx, (lm_data, edit_example, _, _) in enumerate(subset):
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                out = model(lm_tokens, labels=lm_labels)

                loss = out.loss
                total_loss.append(loss)

        return torch.exp(torch.mean(torch.stack(total_loss)))

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
        print("Starting Training")

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
                    print(f"Unable to locate edit on TS {train_step}")
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
                if train_step % 5 == 0:
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
                    print("Reached max iterations")
                    break

            # if (train_step % 1000 == 0) & (not self.config.debug):
            #     self.validateEditTraining()
        
        self.saveState(self.model, global_iter, final=True, name='editable')
        if self.config.learnable_lr:
            self.saveState(lr_opt, global_iter, final=True, name='lr')


class SelfSampleTrainer(EditTrainer):
    def __init__(self, config, dataloader, model_path=None):
        super().__init__(config, dataloader, model_path)

        self.validation_set = utils.retrieveEditDataloader(
            self.tokenizer,
            bs=1,
            data_loc=self.config.write_loc,
            dataset='validation',
            self_sample=True
        )
        
        self.finetuned = utils.loadTrainedModel(
            f"{self.config.write_loc}/models/finetune/{self.config.ft_model_name}", 
            cache_dir=self.config.write_loc,
            tokenizer=False
        )
        self.finetuned.eval()
        self.finetuned.to(self.device)
        
    def genModelText(self, lm_tokens):
        
        len_lm = lm_tokens.shape[-1]
        edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
        input_ids = lm_tokens[:, :edit_loc]
        input_size = input_ids.size()[-1]
        
        print("generating")
        output_sequence = self.finetuned.generate(
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
    
    def processLMData(self, lm_data, edit_example):
        lm_tokens, lm_mask = lm_data
        lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
        lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

        edit_tokens, edit_mask = edit_example
        # remove left padding
        edit_tokens = edit_tokens.squeeze(0)
        indices = edit_tokens != 50256
        edit_tokens = edit_tokens[indices].unsqueeze(0)
        edit_mask = edit_mask.squeeze(0)
        edit_mask = edit_mask[indices].unsqueeze(0)

        edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
        edit_loc = edit_tokens.shape[-1] - 5 - 1  # minus 1 for newline token
        edit_locs = torch.tensor([edit_loc + i for i in range(5)])
        edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
        gold_tokens = edit_tokens[:, edit_locs]

        edit_labels = edit_labels.to(self.device)
        edit_tokens, edit_mask = edit_tokens.to(self.device), edit_mask.to(self.device)

        gold_tokens = gold_tokens.cpu()

        return lm_tokens, lm_mask, lm_labels, edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens

    def validateSelfSampleTraining(self):
        self.model.eval()

        for ds in ['train', 'val']:
            data = self.validation_set if ds == 'val' else self.data

            ppl_pre_hist = []
            ppl_post_hist = []
            ll_change_hist = []
            loss_hist = []

            indices = [idx % len(data) for idx in range(self.val_iter, self.val_iter + 10)]  # select 10 elements
            subset = Subset(dataloader.dataset, indices)
            for batch_idx, (lm_data, edit_example, _, _) in enumerate(subset):
                lm_tokens, lm_mask, lm_labels, edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens = self.processLMData(lm_data, edit_example)

                orig_ppl = self.perplexity(self.model, data)
                model_out, logit_hist, ll_change, loss = performOneEdit(
                    self.model,
                    self.lrs,
                    edit_tokens, 
                    edit_mask, 
                    edit_labels,
                    edit_locs - 1, 
                    gold_tokens, 
                    n_edit_steps=1
                )
                new_ppl = self.perplexity(model_out, data)

                ppl_pre_hist.append(orig_ppl.cpu())
                ppl_post_hist.append(new_ppl.cpu())
                ll_change_hist.append(ll_change)
                loss_hist.append(loss.detach().cpu())

            metrics = {
                f'ppl_pre_{ds}': np.mean(ppl_pre_hist),
                f'ppl_post_{ds}': np.mean(ppl_post_hist),
                f'll_change_{ds}': np.mean(ll_change_hist),
                f'loss/eval_{ds}': np.mean(loss_hist),
            }
            self.wandb_log(self.val_iter * 100, metrics)

        self.val_iter += 1

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
        
        global_iter = 0
        print("Starting Training")
        
        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.model.transformer.h[-3:].parameters()
            ]
        lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)

        skip_count = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            for train_step, (lm_data, edit_example, _, _) in enumerate(self.data):

                lm_tokens, lm_mask, lm_labels, edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens = self.processLMData(lm_data, edit_example)

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
                if train_step % 5 == 0:
                    opt.step()
                    opt.zero_grad()
                
                    if self.config.learnable_lr:
                        lr_opt.step()
                        lr_opt.zero_grad()
                
                global_iter += 1
                
                loss_dict = {
                    "loss/base": l_base, "loss/edit": l_edit, 
                    "loss/loc": l_loc, "loss/train": total_loss
                    }
                self.echo(train_step, **loss_dict)
                loss_dict.update({f"lr/lr{i}":lr.data.item() for i, lr in enumerate(self.lrs)})
                self.wandb_log(global_iter, loss_dict)
                self.saveState(self.model, global_iter, name="self_sample")
                if self.config.learnable_lr:
                    self.saveState(self.lrs, global_iter, name='lr')
                if global_iter >= self.config.max_iter:
                    print("Reached max iterations")
                    break
            
                if (train_step % 100 == 0) & (not self.config.debug):
                    self.validateSelfSampleTraining()
        
        self.saveState(self.model, global_iter, final=True, name="self_sample")
        if self.config.learnable_lr:
            self.saveState(self.lrs, global_iter, final=True, name='lr')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--self_sample', action='store_true')
    parser.add_argument('--bs', default=1, type=int)
    args = parser.parse_args()
    
    loc = utils.sailPreprocess()
    tokenizer = utils.loadTokenizer(cache_dir=loc)

    if not (args.editable or args.self_sample):
        dataloader = utils.wikiDataloader(
            tokenizer,
            bs=args.bs,
            data_loc=loc,
            dataset='train',
            shuffle=False,
            max_length=200,
            min_length=20
        )
    else:
        dataloader = utils.retrieveEditDataloader(
            tokenizer,
            data_loc=loc,
            bs=args.bs,
            dataset='train',
            self_sample=args.self_sample
        )

    if args.editable:
        config = EditConfig()
        config.write_loc = loc
        config.bs = args.bs
        trainer = EditTrainer(config, dataloader)
    
    elif args.finetune:
        config = TrainConfig()
        config.write_loc = loc
        config.bs = args.bs
        trainer = BaseTrainer(config, dataloader)
    
    elif args.self_sample:
        config = SelfSampleConfig()
        config.write_loc = loc
        config.bs = args.bs
        trainer = SelfSampleTrainer(config, dataloader, tokenizer)
    
    else:
        raise AttributeError("Must specify train arg")

    trainer.run()
