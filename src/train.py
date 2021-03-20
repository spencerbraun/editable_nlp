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

from utils import loadOTSModel, retrieveDataloader, locateEntityEdit

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TrainConfig:
    def __init__(self):
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e4
        self.cedit = 1
        self.cloc = 1

        self.debug = False
        self.model_save_pt = 2500
        self.model_dir = '../models/finetune'

        self.hypers = {
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'epochs': self.epochs,
            'n_edit_steps': self.n_edit_steps,
            'cedit': self.cedit,
            'cloc': self.cloc
        }

class EditConfig:
    def __init__(self):
        self.inner_lr = 1e-3
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e4
        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 1

        self.debug = False
        self.model_save_pt = 2500
        self.model_dir = '../models/finetune'

        self.hypers = {
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'epochs': self.epochs,
            'n_edit_steps': self.n_edit_steps,
            'cedit': self.cedit,
            'cloc': self.cloc
        }


class BaseTrainer:
    def __init__(self, config, dataloader):

        #configs
        self.config = config

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

        self.epoch = 0

    def saveModel(self, model, train_step):
        
        if (train_step > 0) & (train_step % self.config.model_save_pt == 0):
            torch.save(
                model.state_dict(), 
                self.modelpath("gpt2", self.epoch, train_step)
                )

    def echo(self, train_step, l_base):
        print((
                f"Epoch: {self.epoch}; TrainStep {train_step}; ",
                f"finetune_loss {l_base}"
            )) 

    def tensorBoard(self, step, l_base):
        writer.add_scalar("Lbase", l_base, step)

    def run(self):
            
        writer = SummaryWriter()
        if not self.debug:
            torch.save(self.config.hypers, self.hyperspath)

        model.train()
        model.to(self.device)
        opt = torch.optim.Adam(
            model.parameters(), 
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
                self.tensorBoard(global_iter)
                self.saveModel(model, train_step)

        self.saveModel(model, train_step)
        writer.flush()



class EditTrainer(BaseTrainer):
    def __init__(self, config):
         super().__init__(config) 


    def echo(self, epoch, train_step, l_base, l_edit, l_loc, total_loss):
        print((
                f"Epoch: {self.epoch}; TrainStep {train_step}; ",
                f"L_edit {l_edit} L_base {l_base} L_loc {l_loc}; ",
                f"Total Loss {total_loss}"
            )) 

    def tensorBoard(self, step, l_base, l_edit, l_loc, total_loss):
        writer.add_scalar("Lbase", l_base, step)
        writer.add_scalar("Ledit", l_edit, step)
        writer.add_scalar("Lloc", l_loc, step)
        writer.add_scalar("total_loss", total_loss, step)

    
    def run(self):
        writer = SummaryWriter()
        if not self.debug:
            torch.save(self.config.hypers, self.hyperspath)

        model.train()
        model.to(self.device)
        opt = torch.optim.Adam(
            model.parameters(), 
            self.config.outer_lr
            )
        
        global_iter = 0
        print("Starting Fine-tuning")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

            edit_tokens, edit_mask = edit_example
            
            ent_tokens = ent[0].flatten()
            ent_tokens = ent_tokens[ent_tokens != 50256]
            edit_locs = locateEntityEdit(edit_tokens, ent_tokens)
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
                self.tensorBoard(global_iter, l_base, l_edit, l_loc, total_loss)
                self.saveModel(model, train_step)

            if train_step % 1000 == 0:
                valid_iter = validateEditTraining(
                    model, 
                    validation_set, 
                    writer, 
                    valid_iter
                    )
                
        self.saveModel(model, train_step)
        writer.flush()


def editableTrainLoop(
    model, 
    dataloader, 
    validation_set,
    epochs, 
    n_edit_steps=1, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):

    
    writer = SummaryWriter()
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    errpath  = f"errors/errors_{timestamp}"
    os.mkdir(errpath)
    hyperspath = f"../models/hypers.{timestamp}"
    hypers = {
        'inner_lr': lr,
        'outer_lr': 1e-5,
        'n_edit_steps': n_edit_steps,
        'cedit': cedit,
        'cloc': cloc
    }
    torch.save(hypers, savepath)
    
    total_epochs = epochs
    model.train()
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    global_iter = 0
    valid_iter = 0
    print("starting training")

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

            edit_tokens, edit_mask = edit_example
            
            ent_tokens = ent[0].flatten()
            ent_tokens = ent_tokens[ent_tokens != 50256]
            edit_locs = locateEntityEdit(edit_tokens, ent_tokens)
            if edit_locs.size == 0:
                print(f"Unable to locate edit on TS {train_step}")
                torch.save(edit_tokens, f"{errpath}/edit_tokens_{train_step}")
                torch.save(ent_tokens, f"{errpath}/ent_tokens_{train_step}")
                continue
            
            edit_labels = torch.zeros(edit_tokens.shape, dtype=torch.long) - 100
            edit_labels[:, edit_locs] = edit_tokens[:, edit_locs]
            edit_labels = edit_labels.to(DEVICE)
            edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)
            
            inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)
            # inner_opt = torch.optim.SGD(model.parameters(), lr=lr)
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

                print((
                    f"Epoch: {epoch}; TrainStep {train_step}; ",
                    f"L_edit {l_edit} L_base {l_base} L_loc {l_loc}; ",
                    f"Total Loss {total_loss}"
                )) 

                writer.add_scalar("Lbase", l_base, global_iter)
                writer.add_scalar("Ledit", l_edit, global_iter)
                writer.add_scalar("Lloc", l_loc, global_iter)
                writer.add_scalar("total_loss", total_loss, global_iter)
            

            if (train_step > 0) & (train_step % 1000 == 0):
                valid_iter = validateEditTraining(
                    model, 
                    validation_set, 
                    writer, 
                    valid_iter
                    )
            if (train_step > 0) & (train_step % 2000 == 0):
                torch.save(
                    model.state_dict(), 
                    f"../models/model_epoch{epoch}_ts{train_step}.{timestamp}"
                    )  
            if (train_step > 0) & (train_step % 5000 == 0):
                torch.save(
                    fmodel.state_dict(), 
                    f"../models/fmodel_epoch{epoch}_ts{train_step}.{timestamp}"
                    ) 

    torch.save(model.state_dict(), f"../models/model_epoch_FINAL.{timestamp}")
    writer.flush()

def finetuneBaseline(
    model, 
    dataloader, 
    epochs, 
    lr=0.01
    ):

    writer = SummaryWriter()
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    total_epochs = epochs
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    hypers = {
        'inner_lr': lr,
        'outer_lr': 1e-5,
        'epochs': total_epochs
    }
    savepath = f"../models/finetune/hypers.{timestamp}"
    torch.save(hypers, savepath)
    
    model.to(DEVICE)
    global_iter = 0
    print("starting training")

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, _, _) in enumerate(dataloader):
            
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

            # accumulate grads 
            if train_step % 5 == 0:
                opt.step()
                opt.zero_grad()
            
            global_iter += 1

            print((
                f"Epoch: {epoch}; TrainStep {train_step}; ",
                f"finetune_loss {l_base}"
            )) 

            writer.add_scalar("finetune_loss", l_base, global_iter)
    
            if (train_step > 0) & (train_step % 2500 == 0):
                torch.save(
                    model.state_dict(), 
                    f"../models/finetune/gpt2_epoch{epoch}_ts{train_step}.{timestamp}"
                    )

    torch.save(model.state_dict(), f"../models/finetune/gpt2_epoch_FINAL.{timestamp}")
    writer.flush()


def validateEditTraining(model, validation_set, writer, start=0):

    model.eval()
    global_iter = start
    for train_step, (lm_data, _, _) in enumerate(validation_set):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            
            base_out = model(
                lm_tokens, 
                attention_mask=lm_mask,
                labels=lm_labels
            )
            writer.add_scalar("val_lbase", base_out.loss, global_iter)
            global_iter +=1
    
    model.train()
    return global_iter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    
    model, tokenizer = loadOTSModel()
    dataloader = retrieveDataloader(
        tokenizer, 
        bs=1, 
        dataset='train'
    )
    validation_set = retrieveDataloader(
        tokenizer, 
        bs=15, 
        dataset='valid',
        max_obs=1000,
        shuffle=True
    )
    
    if args.editable:
        editableTrainLoop(
            model, 
            dataloader, 
            validation_set,
            epochs=1,
            n_edit_steps=1, 
            cedit=0.1, 
            cloc=0.1, 
            lr=1e-3
        )
    
    if args.finetune:
        finetuneBaseline(
            model, 
            dataloader, 
            epochs=1, 
            lr=1e-5
        )
