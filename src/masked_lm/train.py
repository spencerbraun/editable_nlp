import os
import argparse
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

import transformers
import higher
import wandb

import utils
from config import TrainConfig

class T5Trainer:
    def __init__(self, config, train, validation, model_path=None):

        #configs
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_dir = f'{self.config.write_loc}/models/t5'
        self.model, self.tokenizer = (
            utils.loadT5Model(cache_dir=self.config.write_loc) if not model_path else 
            utils.loadTrainedModel(model_path, cache_dir=self.config.write_loc)
            )
        self.model.to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), 
            self.config.outer_lr
            )

        #outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.statepath = (
            lambda model, epoch, step: 
            f"{self.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
            )

        self.train = train
        self.validation = validation

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
        self.train_iter = 0
        self.valid_iter = 0

    def saveState(self, state_obj, train_step, name="T5_finetune", final=False):
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

    def train_step(self):
        
        print("Train Steps")
        self.model.train()
        for train_step, noised_batch in enumerate(self.train):
                
                tokens, labels = noised_batch
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                
                base_out = self.model(
                    tokens,
                    labels=labels
                )
                l_base = base_out.loss
                l_base.backward()

                if train_step % 5 == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                
                if train_step % 2000 == 0:
                    self.valid_step()

                self.echo(self.train_iter, **{"loss/base": l_base})
                self.wandb_log(self.train_iter, {"loss/base": l_base})
                self.saveState(self.model, self.train_iter, "T5_finetune")
                self.train_iter += 1
        

    def valid_step(self):
        
        print("Validation Steps")
        self.model.eval()
        with torch.no_grad():
            for train_step, noised_batch in enumerate(self.validation):
                    
                tokens, labels = noised_batch
                tokens, labels = tokens.to(self.device), labels.to(self.device)
                
                base_out = self.model(
                    tokens,
                    labels=labels
                )
                l_base = base_out.loss
                l_base.backward()
         
            self.echo(self.valid_iter, **{"Val\tLoss:": l_base})
            self.wandb_log(self.valid_iter, {
                "loss/val_loss": l_base
                })
            self.valid_iter += 1
                

    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.train_step()

        self.saveState(self.model, self.train_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    loc = utils.sailPreprocess()
    tokenizer = utils.loadTokenizer(cache_dir=loc)

    dl = utils.MaskedLMDataloader(
        'lama'
        tokenizer,
        loc=loc,
        bs=args.bs,
        pct=100
        shuffle=True
    )
    
    config = TrainConfig()
    config.write_loc = loc
    config.bs = args.bs
    
    trainer = T5Trainer(config, dl.train, dl.validation)
    trainer.run()
