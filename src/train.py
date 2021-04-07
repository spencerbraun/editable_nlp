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
import higher
from torch.utils.tensorboard import SummaryWriter

import utils
from config import TrainConfig, EditConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseTrainer:
    def __init__(self, config, dataloader, model_path=None):

        #configs
        self.config = config
        self.model, self.tokenizer = (
            utils.loadOTSModel(cache_dir=self.config.write_loc) if not model_path else 
            utils.loadTrainedModel(model_path, cache_dir=self.config.write_loc)
            )

        #outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.config.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}"
        self.modelpath = (
            lambda model, epoch, step: 
            f"{self.config.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
            )

        self.data = dataloader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer = (
            SummaryWriter(log_dir=f"{self.config.write_loc}/runs") 
            if not self.config.debug else None
            )
        self.epoch = 0

    def saveModel(self, model, train_step, name="model"):
        if not self.config.debug:
            if (train_step > 0) & (train_step % self.config.model_save_pt == 0):
                torch.save(
                    self.model.state_dict(), 
                    self.modelpath(name, self.epoch, train_step)
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
            
            for train_step, (lm_data, _, _) in enumerate(self.data):
                
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                
                base_out = model(
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
                self.echo(train_step, **{"l_base": l_base})
                self.tensorBoard(global_iter, **{"l_base": l_base})
                self.saveModel(self.model, train_step, "gpt2")

        
        self.saveModel(self.model, train_step)
        self.writer.flush()



class EditTrainer(BaseTrainer):
    def __init__(self, config, dataloader, model_path=None):
        super().__init__(config, dataloader, model_path=None) 
        self.validation_set = utils.retrieveDataloader(
            self.tokenizer, 
            bs=15, 
            data_loc=self.config.write_loc,
            dataset='valid',
            max_obs=1000,
            shuffle=True
        )
        self.val_iter = 0

    def validateEditTraining(self):
        self.model.eval()
        for train_step, (lm_data, _, _) in enumerate(self.validation_set):
                
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                
                base_out = self.model(
                    lm_tokens, 
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                self.writer.add_scalar("val_lbase", base_out.loss, self.val_iter)
                self.val_iter +=1
        
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

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            for train_step, (lm_data, edit_example, ent) in enumerate(self.data):
            
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

                edit_tokens, edit_mask = edit_example
                
                ent_tokens = ent[0].flatten()
                ent_tokens = ent_tokens[ent_tokens != 50256]
                edit_locs = utils.locateEntityEdit(edit_tokens, ent_tokens)
                if edit_locs.size == 0:
                    print(f"Unable to locate edit on TS {train_step}")
                    torch.save(edit_tokens, f"{self.errpath}/edit_tokens_{train_step}")
                    torch.save(ent_tokens, f"{self.errpath}/ent_tokens_{train_step}")
                    continue
                
                edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
                edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
                edit_labels = edit_labels.to(self.device)
                edit_tokens, edit_mask = edit_tokens.to(self.device), edit_mask.to(self.device)
                
                inner_opt = torch.optim.SGD(
                    self.model.transformer.h[-3:].parameters(), 
                    lr=self.config.inner_lr
                    )

                with higher.innerloop_ctx(
                    self.model, 
                    inner_opt, 
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
                
                global_iter += 1
                
                loss_dict = {
                    "l_base": l_base, "l_edit": l_edit, 
                    "l_loc": l_loc, "total": total_loss
                    }
                self.echo(train_step, **loss_dict)
                self.tensorBoard(global_iter, **loss_dict)
                self.saveModel(self.model, train_step)

            if (train_step % 1000 == 0) & not self.config.debug:
                self.validateEditTraining()

        
        self.saveModel(self.model, train_step)
        self.writer.flush()


class SelfSampleTrainer(EditTrainer):
    def __init__(self, config, dataloader, model_path=None):
        super().__init__(config, dataloader, model_path) 
        
        self.finetuned = utils.loadTrainedModel(
            f"{self.config.write_loc}/models/finetune/gpt2_epoch0_ts10000.20210310.18.03.1615401990", 
            cache_dir=self.config.write_loc,
            tokenizer=False
        )
        self.finetuned.to(self.device)
        
    def genModelText(self, lm_tokens, edit_locs):
        

        input_ids = lm_tokens[:, :edit_locs.min()]
        input_size = input_ids.size()[-1]
        
        self.finetuned.eval()
        print("generating")
        output_sequence = self.finetuned.generate(
            input_ids=input_ids,
            max_length=input_size + 5,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=10,
        )

        edit_tokens = random.choice(output_sequence).unsqueeze(0)
        edit_mask = torch.ones(edit_tokens.shape, dtype=torch.long)
        edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
        edit_labels[:, input_size:] = edit_tokens[:, input_size:]
        edit_labels = edit_labels.to(self.device)
        edit_tokens, edit_mask = edit_tokens.to(self.device), edit_mask.to(self.device)

        return edit_tokens, edit_mask, edit_labels

    
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
        
        lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.model.parameters()
            ]
        lr_opt = torch.optim.Adam(lrs, lr=1e-2)

        skip_count = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            for train_step, (lm_data, edit_example, ent) in enumerate(self.data):
            
                lm_tokens, lm_mask = lm_data
                lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
                lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
                
                ent_tokens = ent[0].flatten()
                ent_tokens = ent_tokens[ent_tokens != 50256]
                edit_locs = utils.locateEntityEdit(edit_example[0], ent_tokens)
                if edit_locs.size == 0 or edit_locs.min() < 10:
                    skip_count += 1
                    continue
                if train_step % 50 == 0:
                    print(f"SKIPCOUNT: {skip_count}")
                    
                edit_tokens, edit_mask, edit_labels = self.genModelText(lm_tokens, edit_locs)

                inner_opt = torch.optim.SGD(
                    self.model.transformer.h[-3:].parameters(), 
                    lr=self.config.inner_lr
                    )

                param_groups = [{'params': p, 'lr': None} for p in self.model.parameters()]
                inner_opt = torch.optim.SGD(param_groups)
        
                with higher.innerloop_ctx(
                    self.model, 
                    inner_opt, 
                    override={'lr': lrs},
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
                
                # step inner loop lr
                lr_opt.step()
                lr_opt.zero_grad()
                
                global_iter += 1
                
                loss_dict = {
                    "l_base": l_base, "l_edit": l_edit, 
                    "l_loc": l_loc, "total": total_loss
                    }
                self.echo(train_step, **loss_dict)
                self.tensorBoard(global_iter, **loss_dict)
                self.saveModel(self.model, train_step)
        
        self.saveModel(self.model, train_step)
        self.writer.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--self_sample', action='store_true')
    parser.add_argument('--data_loc', default='..')
    args = parser.parse_args()

    model_del, tokenizer = utils.loadOTSModel(cache_dir=data_loc)
    tokenizer.pad_token = tokenizer.eos_token
    del model_del

    dataloader = utils.retrieveDataloader(
        tokenizer, 
        data_loc=args.data_loc,
        bs=1, 
        dataset='train'
    )

    if args.editable:
        trainer = EditTrainer(EditConfig(), dataloader)
    
    elif args.finetune:
        trainer = BaseTrainer(TrainConfig(), dataloader)
    
    elif args.self_sample:
        trainer = SelfSampleTrainer(EditConfig(), dataloader)
    
    else:
        raise AttributeError("Must specify train arg")

    trainer.run()
