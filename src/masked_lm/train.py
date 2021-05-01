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

import data
import utils
from config import TrainConfig

import sys
sys.path.insert(0, '../..')
from src.train import EditableTrainer


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
            
            if (train_step % 5 == 0) & (train_step != 0):
                self.opt.step()
                self.opt.zero_grad()
                
            if (train_step % 2000 == 0) & (train_step != 0):
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
        
        self.model = utils.loadTrainedModel(
            f"{self.config.write_loc}/models/finetune/{self.config.ft_model_name}", 
            cache_dir=self.config.write_loc,
            tokenizer=False
        )
        self.model.eval()
        self.model.to(self.device)

        self.model_type = self.config.model_type

        if self.config.split_params:
            ConditionedLinear.add_conditioners(self.model)
        
    def genModelText(self, lm_tokens):
        
        len_lm = lm_tokens.shape[-1]
        edit_loc = max(random.randint(int(len_lm*0.6), int(len_lm*0.9)), 15)
        input_ids = lm_tokens[:, :edit_loc]
        input_size = input_ids.size()[-1]
        
        print("generating")
        output_sequence = self.model.generate(
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

    def processMaskedLMData(self, masked_sentence, template, obj, edit_obj):
        il = self.config.inner_loop
        il_select = (np.random.randint(0,2) if il == 'random' else 1 if il == 'sentence' else 0)
        lm_data = masked_sentence if il_select == 0 else template
        edit_data = masked_sentence if il_select == 1 else template 

        lm_tokens, lm_mask = lm_data
        lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)

        lm_labels, _ = obj
        lm_labels = lm_labels.to(self.device)

        edit_tokens_batch, edit_mask_batch = tuple(zip(*edit_example))
        edit_tokens, edit_mask = (
            edit_tokens.to(self.device), 
            edit_mask.to(self.device)
            )

        edit_labels, _ = edit_obj
        edit_labels = edit_labels.to(self.device)
        
        # List of tuples
        edit_batch = [_process_edit_tokens(et, em) for (et, em) in
                      zip(edit_tokens_batch, edit_mask_batch)]

        # Tuple of lists
        edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens = tuple(zip(*edit_batch))

        return (
            lm_tokens, lm_mask, lm_labels, 
            edit_tokens, edit_mask, edit_labels, 
            edit_locs, gold_tokens
            )

    
    def processLMData(self, lm_data, edit_example):
        lm_tokens, lm_mask = lm_data[0]
        lm_tokens, lm_mask = lm_tokens.to(self.device), lm_mask.to(self.device)
        lm_labels = lm_tokens.masked_fill(lm_mask == 0, -100)

        edit_tokens_batch, edit_mask_batch = tuple(zip(*edit_example))

        # remove left padding
        def _process_edit_tokens(edit_tokens, edit_mask):
            edit_tokens = edit_tokens.squeeze(0)
            indices = edit_tokens != self.tokenizer.pad_token_id
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

            return edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens

        # List of tuples
        edit_batch = [_process_edit_tokens(et, em) for (et, em) in
                      zip(edit_tokens_batch, edit_mask_batch)]

        # Tuple of lists
        edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens = tuple(zip(*edit_batch))

        return lm_tokens, lm_mask, lm_labels, edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens

    def processData(self):

        return self.processLMData if self.model_type == 'gpt2' else self.processMaskedLMData

    def validateSelfSampleTraining(self):
        self.model.eval()

        for ds in ['train', 'val']:
            data = self.validation_set if ds == 'val' else self.data

            ppl_pre_hist = []
            ppl_post_hist = []
            ll_change_hist = []
            loss_hist = []

            indices = [idx % len(data) for idx in range(self.val_iter, self.val_iter + 10)]  # select 10 elements
            subset = Subset(data.dataset, indices)
            for batch_idx, (lm_data, edit_example, _, _) in enumerate(subset):
                lm_tokens, lm_mask, lm_labels, edit_tokens, edit_mask, edit_labels, edit_locs, gold_tokens = self.processLMData(lm_data, edit_example)

                orig_ppl = self.perplexity(self.model, data)
                model_out, logit_hist, ll_change, loss = performOneEdit(
                    self.model,
                    self.lrs,
                    edit_tokens[0], 
                    edit_mask[0], 
                    edit_labels[0],
                    edit_locs[0] - 1, 
                    gold_tokens[0], 
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

                # Compute base loss
                base_out = self.model(
                    lm_tokens, 
                    attention_mask=lm_mask,
                    labels=lm_labels
                )
                l_base = base_out.loss
                l_base.backward()

                # Cache the current params and grads since we're going to modify the model during
                #  the edit process
                p_cache = {}
                for n, p in self.model.named_parameters():
                    p_cache[n] = (p.data.detach().clone(), p.grad.data.detach().clone())

                for edit_example_idx in range(self.config.n_edits):
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
                        if self.config.split_params:
                            fmodel.set_editing(True)

                        for edit_step in range(self.config.n_edit_steps):
                            loss = fmodel(
                                edit_tokens, 
                                attention_mask=edit_mask,
                                labels=edit_labels
                            ).loss
                            diffopt.step(loss)

                        if self.config.split_params:
                            fmodel.set_editing(False)

                        edit_out = fmodel(
                            edit_tokens, 
                            attention_mask=edit_mask,
                            labels=edit_labels
                        )
                        l_edit = edit_out.loss

                        edited_base_out = fmodel(
                            lm_tokens, 
                            attention_mask=lm_mask,
                            labels=lm_labels
                        )

                        l_loc = (
                            F.softmax(base_out.logits.detach(), dim=-1) *
                            (
                                F.log_softmax(base_out.logits.detach(), dim=-1) - 
                                F.log_softmax(edited_base_out.logits, dim=-1)
                            )).sum(-1).mean()

                        total_edit_loss = (
                            self.config.cloc * l_loc  + 
                            self.config.cedit * l_edit
                        ) / self.config.n_edits
                        total_edit_loss.backward()

                        for fp, p in zip(fmodel.parameters(), self.model.parameters()):
                            p.data[:] = fp.data.detach().clone()

                    # It only makes sense to train more than one edit if we've split the params
                    if not self.config.split_params:
                        break

                # merge gradients from edit training and restore pre-edit parameters
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        p.grad += p_cache[n][1]
                    else:
                        p.grad = p_cache[n][1]

                    p.data = p_cache[n][0]

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
        shuffle=False
        #max_valid_len=2000
    )
    train = dl.train
    validation = dl.validation
    
    config = TrainConfig()
    config.write_loc = loc
    config.bs = args.bs
    
    trainer = T5Trainer(config, train, validation)
    trainer.run()
