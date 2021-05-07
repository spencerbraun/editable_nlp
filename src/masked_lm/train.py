import argparse
import sys
from datetime import datetime

import numpy as np
import torch

import transformers
import higher
import wandb

import data
from config import TrainConfig

sys.path.append('../')
import utils



class T5Trainer:
    def __init__(self, config, train, validation, model_path=None):

        #configs
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_dir = f'{self.config.write_loc}/models/finetune'
        self.model, self.tokenizer = utils.loadOTSModel(
            name='t5-small',cache_dir=self.config.write_loc
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
                project='patchable_masked',
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

    def saveState(self, state_obj, step, name="T5_finetune", final=False):
        if not self.config.debug:
            out_obj = state_obj.state_dict() if hasattr(state_obj, "state_dict") else state_obj
            if final:
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, 9999)
                    )
            elif (step > 0) & (step % self.config.model_save_pt == 0):
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, step)
                    )

    def echo(self, step, **kwargs):
        if not self.config.silent:
            print((
                    f"Epoch: {self.epoch}; Iter {step}; ",
                    f"; ".join([f"{key} {val}" for key,val in kwargs.items()])
                )) 

    def tensorBoard(self, step, **kwargs):
        if not self.config.debug:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, step)
    
    def wandb_log(self, step, row):
        if not self.config.debug:
            wandb.log(row)

    def train_step(self):
        print("Train Steps")
        self.model.train()
        
        for train_step, (sentence, label) in enumerate(self.train):

            sent_toks, sent_mask = sentence
            label_toks, label_mask = label
            sent_toks, sent_mask = sent_toks.to(self.device), sent_mask.to(self.device)
            label_toks, label_mask = label_toks.to(self.device), label_mask.to(self.device)

            if sent_toks.shape[-1] > 500:
                print(f"Sequence {train_step} too long: {sent_toks.shape[-1]}. Skipping...")
                continue
            base_out = self.model(
                 sent_toks, 
                 attention_mask=sent_mask,
                 labels=label_toks
            )
            
            l_base = base_out.loss
            l_base.backward()
            
            if (self.train_iter % 5 == 0) & (self.train_iter != 0):
                self.opt.step()
                self.opt.zero_grad()
                
            if (self.train_iter % 2000 == 0) & (self.train_iter != 0):
                self.valid_step()
            
            self.echo(self.train_iter, **{"loss/base": l_base})
            self.wandb_log(self.train_iter, {"loss/base": l_base})
            self.saveState(self.model, self.train_iter, "T5_finetune")
            self.train_iter += 1
            
            if self.train_iter > self.config.max_iter:
                break
        

    def valid_step(self):
        
        print("Validation Steps")
        self.model.eval()
        with torch.no_grad():
            for valid_step, (sentence, label) in enumerate(self.validation):
                    
                sent_toks, sent_mask = sentence
                label_toks, label_mask = label
                sent_toks, sent_mask = sent_toks.to(self.device), sent_mask.to(self.device)
                label_toks, label_mask = label_toks.to(self.device), label_mask.to(self.device)

                if sent_toks.shape[-1] > 500:
                    print(f"Sequence {valid_step} too long: {sent_toks.shape[-1]}. Skipping...")
                    continue
                    
                base_out = self.model(
                 sent_toks, 
                 attention_mask=sent_mask,
                 labels=label_toks
                )

                l_base = base_out.loss
         
                self.echo(self.valid_iter, **{"Val Loss:": l_base})
                self.wandb_log(self.valid_iter, {"loss/val_loss": l_base})
                self.valid_iter += 1

        self.model.train()
                

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

    dl = data.MaskedLMDataloader(
        'lama',
        loc=loc,
        bs=args.bs,
        pct=30,
        shuffle=True,
        mode='finetune',
        max_val_len=2000
    )
    train = dl.train
    validation = dl.validation
    
    config = TrainConfig()
    config.write_loc = loc
    config.bs = args.bs
    
    trainer = T5Trainer(config, train, validation)
    trainer.run()
