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

import sys
sys.path.append("/juice/scr/spencerb/editable_nlp/src")
import utils
from utils import loadOTSModel, retrieveEditDataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def editableTrainLoop(
    model, 
    dataloader, 
    validation_set,
    epochs, 
    loc,
    n_edit_steps=1, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):

    
    
    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    writer = SummaryWriter(log_dir=f"{loc}/runs/editable_orig_{timestamp}") 
    # errpath  = f"errors/errors_{timestamp}"
    # os.mkdir(errpath)
    hyperspath = f"{loc}/models/hypers.{timestamp}"
    hypers = {
        'inner_lr': lr,
        'outer_lr': 1e-5,
        'n_edit_steps': n_edit_steps,
        'cedit': cedit,
        'cloc': cloc
    }
    torch.save(hypers, hyperspath)
    
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
            edit_locs = utils.locateSubset(edit_tokens, ent_tokens)
            if edit_locs.size == 0:
                print(f"Unable to locate edit on TS {train_step}")
                # torch.save(edit_tokens, f"{errpath}/edit_tokens_{train_step}")
                # torch.save(ent_tokens, f"{errpath}/ent_tokens_{train_step}")
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

                writer.add_scalar("l_base", l_base, global_iter)
                writer.add_scalar("l_edit", l_edit, global_iter)
                writer.add_scalar("l_loc", l_loc, global_iter)
                writer.add_scalar("total", total_loss, global_iter)
            

            #if (train_step > 0) & (train_step % 1000 == 0):
                #valid_iter = validateEditTraining(
                    #model, 
                    #validation_set, 
                    #writer, 
                    #valid_iter
                    #)
            if (train_step > 0) & (train_step % 2000 == 0):
                torch.save(
                    model.state_dict(), 
                    f"{loc}/models/model_epoch{epoch}_ts{train_step}.{timestamp}"
                    )  
            # if (train_step > 0) & (train_step % 5000 == 0):
            #     torch.save(
            #         fmodel.state_dict(), 
            #         f"{loc}/models/fmodel_epoch{epoch}_ts{train_step}.{timestamp}"
            #         ) 

    torch.save(model.state_dict(), f"{loc}/models/model_epoch_FINAL.{timestamp}")
    writer.flush()

def genModelText(finetuned, lm_tokens, edit_locs):
        

    input_ids = lm_tokens[:, :edit_locs.min()]
    input_size = input_ids.size()[-1]
    
    finetuned.eval()
    print("generating")
    output_sequence = finetuned.generate(
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
    edit_labels = edit_labels.to(DEVICE)
    edit_tokens, edit_mask = edit_tokens.to(DEVICE), edit_mask.to(DEVICE)

    return edit_tokens, edit_mask, edit_labels


def editableSelfSampling(
    model, 
    dataloader, 
    validation_set,
    epochs, 
    n_edit_steps=1, 
    cedit=1, 
    cloc=1, 
    lr=0.01
    ):
    finetuned = utils.loadTrainedModel(
            "../models/finetune/gpt2_epoch0_ts10000.20210310.18.03.1615401990", 
            tokenizer=False
        )
    finetuned.to(DEVICE)
    
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

    lrs = [
            torch.nn.Parameter(torch.tensor(inner_lr)) 
            for p in model.parameters()
            ]
    lr_opt = torch.optim.Adam(lrs, lr=1e-2)

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, edit_example, ent) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(DEVICE), lm_mask.to(DEVICE)
            lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)
            
            ent_tokens = ent[0].flatten()
            ent_tokens = ent_tokens[ent_tokens != 50256]
            edit_locs = utils.locateSubset(edit_example[0], ent_tokens)
            if edit_locs.size == 0 or edit_locs.min() == 0:
                continue
            
            edit_tokens, edit_mask, edit_labels = genModelText(finetuned, lm_tokens, edit_locs)
            
            inner_opt = torch.optim.SGD(
                    model.transformer.h[-3:].parameters(), 
                    lr=inner_lr
                    )
            param_groups = [{'params': p, 'lr': None} for p in model.parameters()]
            inner_opt = torch.optim.SGD(param_groups)
            
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

            lr_opt.step()
            lr_opt.zero_grad()
                
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
    parser.add_argument('--self_sample', action='store_true')
    args = parser.parse_args()

    loc = utils.sailPreprocess()
    
    model, tokenizer = loadOTSModel(cache_dir=loc)
    dataloader = retrieveEditDataloader(
        tokenizer, 
        bs=1, 
        data_loc=loc,
        dataset='train'
    )
    validation_set = retrieveEditDataloader(
        tokenizer, 
        bs=15, 
        data_loc=loc,
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
            loc=loc,
            n_edit_steps=1, 
            cedit=1, 
            cloc=10, 
            lr=1e-3
        )
    
    if args.finetune:
        finetuneBaseline(
            model, 
            dataloader, 
            epochs=1, 
            lr=1e-5
        )
    
    if args.self_sample:
        editableSelfSampling(
            model, 
            dataloader, 
            validation_set,
            epochs=1,
            n_edit_steps=1, 
            cedit=1, 
            cloc=1, 
            lr=1e-2
        )
