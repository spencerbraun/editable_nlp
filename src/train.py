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

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset, list_metrics, load_metric

from data_process import TorchDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loadModel():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def produceDataloader(tokenizer, bs=10):
    writtenFiles = glob.glob("../data/permuted*")
    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    dataset = TorchDataset(list(range(fileIndex)), tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs
    )

    return dataloader


def editableTrainLoop(
    model, 
    dataloader, 
    epochs, 
    n_edit_steps = 10, 
    cedit=0.1, 
    cloc=0.1, 
    lr=0.01
    ):

    writer = SummaryWriter()
    total_epochs = epochs

    inner_opt = torch.optim.SGD(model.transformer.h[-3:].parameters(), lr=lr)

    model.to(device)
    model.train()
    global_iter = 0

    for epoch in range(total_epochs):
        
        for train_step, (lm_data, edit_example) in enumerate(dataloader):
            
            lm_tokens, lm_mask = lm_data
            lm_tokens, lm_mask = lm_tokens.to(device), lm_mask.to(device)
            edit_tokens, edit_mask = edit_example
            edit_tokens, edit_mask = edit_tokens.to(device), edit_mask.to(device)
            
            lm_labels = lm_mask.masked_fill(lm_mask == 0, -100)
            edit_labels = edit_mask.masked_fill(edit_mask == 0, -100) 
            
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
                import ipdb; ipdb.set_trace()
                base_out = model(
                    lm_tokens, 
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                l_base = base_out.loss
                
                edit_out = fmodel(
                    edit_tokens, 
                    attention_mask=edit_mask,
                    labels=edit_labels
                )
                l_edit = edit_out.loss
                l_loc = F.kl_div(
                    edit_out.logits,
                    base_out.logits
                )
                
                total_loss = l_base + cedit * l_edit + cloc * l_loc
                total_loss.backward()
                global_iter += 1
                
                print(f"Epoch: {epoch}; TrainStep {train_step}; L_edit {l_edit} L_base {l_base} L_loc {l_loc}; Total Loss {total_loss}")
                writer.add_scalar("Lbase", l_base, global_iter)
                writer.add_scalar("Ledit", l_edit, global_iter)
                writer.add_scalar("Lloc", l_loc, global_iter)
                writer.add_scalar("total_loss", total_loss, global_iter)
        import ipdb; ipdb.set_trace()
        if epoch % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
            torch.save(model.state_dict(), f"../models/model_epoch{epoch}.{timestamp}")
            torch.save(fmodel.state_dict(), f"../models/fmodel_epoch{epoch}.{timestamp}")
        
    writer.flush()


if __name__ == "__main__":

    model, tokenizer = loadModel()
    dataloader = produceDataloader(tokenizer, bs=2)
    editableTrainLoop(
        model, 
        dataloader, 
        5, 
        n_edit_steps = 3, 
        cedit=0.1, 
        cloc=0.1, 
        lr=0.01
    )
