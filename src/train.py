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

from utils import loadOTSModel, retrieveDataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    total_epochs = epochs
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    hypers = {
        'inner_lr': lr,
        'outer_lr': 1e-5,
        'n_edit_steps': n_edit_steps,
        'cedit': cedit,
        'cloc': cloc
    }
    savepath = f"../models/hypers.{timestamp}"
    torch.save(hypers, savepath)
    writer.add_hparams(
        {
            'lr_inner': lr, 'lr_outer': 1e-5, 'cedit': cedit, 'cloc': cloc, 
            'nedit_steps': n_edit_steps
            },
        {'placeholder': 0}
        )
    
    model.to(device)
    global_iter = 0
    valid_iter = 0
    print("starting training")

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, edit_example, _) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(device), lm_mask.to(device)
            edit_tokens, edit_mask = edit_example
            edit_tokens, edit_mask = edit_tokens.to(device), edit_mask.to(device)
            
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            edit_labels = edit_tokens.masked_fill(edit_mask == 0, -100) 
            
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
        
            if (train_step > 0) & (train_step % 2000 == 0):
                torch.save(
                    model.state_dict(), 
                    f"../models/model_epoch{epoch}_ts{train_step}.{timestamp}"
                    )
                valid_iter = validateEditTraining(
                    model, 
                    validation_set, 
                    writer, 
                    valid_iter
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
    
    model.to(device)
    global_iter = 0
    print("starting training")

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, _, _) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(device), lm_mask.to(device)
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
            lm_tokens, lm_mask = lm_tokens.to(device), lm_mask.to(device)
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
    parser.add_argument('--editable', help='')
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
        maxobs=100,
        shuffle=True
    )
    
    if args.editable:
        editableTrainLoop(
            model, 
            dataloader, 
            validation_set,
            epochs=1,
            n_edit_steps=1, 
            cedit=1, 
            cloc=1, 
            lr=1e-3
        )
    
    if args.finetune:
        finetuneBaseline(
            model, 
            dataloader, 
            epochs=1, 
            lr=1e-5
        )
