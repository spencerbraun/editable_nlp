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
            utils.loadOTSModel() if not model_path else 
            utils.loadTrainedModel(model_path)
            )

        #outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.config.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"errors/errors_{self.timestamp}"
        self.modelpath = (
            lambda model, epoch, step: 
            f"{self.config.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
            )

        self.data = dataloader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.writer = SummaryWriter()
        self.epoch = 0

    def saveModel(self, model, train_step):
        
        if (train_step > 0) & (train_step % self.config.model_save_pt == 0):
            torch.save(
                self.model.state_dict(), 
                self.modelpath("gpt2", self.epoch, train_step)
                )

    def echo(self, train_step, **kwargs):
        print((
                f"Epoch: {self.epoch}; TrainStep {train_step}; ".
                f"; ".join([f"{key} {val}" for key,val in kwargs.items()])
            )) 

    def tensorBoard(self, step, l_base):
        self.writer.add_scalar("Lbase", l_base, step)

    def run(self):

        if not self.debug:
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
                self.echo(train_step, l_base)
                if not self.config.debug:
                    self.tensorBoard(global_iter, l_base)
                    self.saveModel(model, train_step)

        if not self.config.debug:
            self.saveModel(model, train_step)
        self.writer.flush()



class EditTrainer(BaseTrainer):
    def __init__(self, config, dataloader, model_path=None):
         super().__init__(config, dataloader, model_path=None) 
         self.validation_set = retrieveDataloader(
            tokenizer, 
            bs=15, 
            dataset='valid',
            max_obs=1000,
            shuffle=True
        )
        self.val_iter = 0

    def tensorBoard(self, step, l_base, l_edit, l_loc, total_loss):
        self.writer.add_scalar("Lbase", l_base, step)
        self.writer.add_scalar("Ledit", l_edit, step)
        self.writer.add_scalar("Lloc", l_loc, step)
        self.writer.add_scalar("total_loss", total_loss, step)

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

        if not self.debug:
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
            
            for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            
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
                    model.transformer.h[-3:].parameters(), 
                    lr=self.config.inner_lr
                    )

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
                        diffopt.step(loss)

                    edit_out = fmodel(
                        edit_tokens, 
                        attention_mask=edit_mask,
                        labels=edit_labels
                    )
                    l_edit = edit_out.loss
                    
                    base_out = model(
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
                    
                    total_loss = l_base + cloc * l_loc  + cedit * l_edit 
                    total_loss.backward()

                    # accumulate grads 
                    if train_step % 5 == 0:
                        opt.step()
                        opt.zero_grad()
                    
                    global_iter += 1

                    self.echo(train_step, l_base, l_edit, l_loc, total_loss)
                    if not self.config.debug:
                        self.tensorBoard(global_iter, l_base, l_edit, l_loc, total_loss)
                        self.saveModel(model, train_step)

                if train_step % 1000 == 0:
                    self.validateEditTraining()
        if not self.config.debug:
            self.saveModel(model, train_step)
        self.writer.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    dataloader = retrieveDataloader(
        tokenizer, 
        bs=1, 
        dataset='train'
    )

    if args.editable:
        trainer = EditTrainer(EditConfig, dataloader)
    
    elif args.finetune:
        trainer = BaseTrainer(TrainConfig, dataloader)
    
    else:
        raise AttributeError("Must specify --editable or --finetune")

    trainer.run()
