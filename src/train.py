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

from data_process import TorchDataset
from utils import loadOTSModel, retrieveDataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def editableTrainLoop(
    model, 
    dataloader, 
    epochs, 
    n_edit_steps=1, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):

    writer = SummaryWriter()
    total_epochs = epochs
    
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.to(device)
    
    global_iter = 0
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
            with higher.innerloop_ctx(
                model, 
                inner_opt, 
                copy_initial_weights=False, 
                track_higher_grads=False
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
                l_edit = cedit * edit_out.loss
                l_edit.backward() #TODO: does this make a memory difference
                
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
                l_loc = F.kl_div(
                    edited_base_out.logits,
                    base_out.logits,
                    reduction='batchmean',
                    log_target=True
                )
                
                total_loss = l_base + cloc * l_loc
                total_loss.backward()
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
        
        if epoch % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
            torch.save(model.state_dict(), f"../models/model_epoch{epoch}.{timestamp}")
            torch.save(fmodel.state_dict(), f"../models/fmodel_epoch{epoch}.{timestamp}")

    timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
    torch.save(model.state_dict(), f"../models/model_epoch_FINAL.{timestamp}")
    torch.save(fmodel.state_dict(), f"../models/fmodel_epoch_FINAL.{timestamp}")    
    writer.flush()


if __name__ == "__main__":

    model, tokenizer = loadOTSModel()
    dataloader = retrieveDataloader(
        tokenizer, 
        bs=2, 
        dataset='train', 
        max_obs=10
    )
    editableTrainLoop(
        model, 
        dataloader, 
        2, 
        n_edit_steps=2, 
        cedit=0.1, 
        cloc=0.1, 
        lr=0.01
    )
