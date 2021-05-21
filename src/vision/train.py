import os
import glob
import time
import random
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision.models import resnet18, densenet169

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

import wandb
import vision.utils as utils
from vision.data_process import loadCIFAR, loadImageNet
from vision.evaluate import accuracy, editGenerator, performEdits
from alg.senn_conditional import ConditionalLinearWrapper

eps = np.finfo(np.float32).eps.item()


class BaseTrainer:
    def __init__(self, config, train_set, val_set):

        # config
        self.config = config
        run_dir = os.getcwd()

        self.model_dir = os.path.join(run_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        num_classes = len(train_set.classes)
        load_model = densenet169 if config.model == 'densenet169' else resnet18
        self.model = utils.loadOTSModel(load_model, num_classes, self.config.pretrained, layernorm=self.config.layernorm)
        if not self.config.pretrained and self.config.model_path:
            model_path = os.path.join(self.config.loc, self.config.model_path)
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
        self.original_model = copy.deepcopy(self.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # outfiles
        date, time = run_dir.split('/')[-2:]
        self.timestamp = f"{date}.{time}"
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{run_dir}/errors/errors_{self.timestamp}"
        self.statepath = (
            lambda model, epoch, step: 
            f"{self.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
        )

        self.configure_data(train_set, val_set)

        if not self.config.debug:
            wandb.init(
                project='patchable-cnn',
                entity='patchable-lm',
                config=self.config,
                name=f"{self.config.dataset}/{self.config.model}/{self.config.task}_{self.timestamp}",
                dir=self.config.loc,
            )

    def saveState(self, state_obj, train_step, name="finetune", final=False):
        if not self.config.debug:
            out_obj = state_obj.state_dict() if hasattr(state_obj, "state_dict") else state_obj
            if final:
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, 9999)
                )
            elif train_step % self.config.model_save_pt == 0:
                torch.save(
                    out_obj, 
                    self.statepath(name, self.epoch, train_step)
                )

    def echo(self, train_step, **kwargs):
        if not self.config.silent:
            print((
                f"Epoch: {self.epoch}; TrainStep {train_step}; ",
                "; ".join([f"{key} {val}" for key,val in kwargs.items()])
            ))
    
    def verbose_echo(self, message):
        if not self.config.silent and self.config.verbose:
            print(message)

    def configure_data(self, train_set, val_set):
        self.train_set = train_set
        self.val_set = val_set

    def compute_base_loss(self, model, inputs, labels):
        if self.config.loss == 'cross_entropy':
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

        elif self.config.loss == 'kl':
            logits = model(inputs)
            self.original_model.eval()
            self.original_model.to(self.device)
            with torch.no_grad():
                orig_logits = self.original_model(inputs)

            loss = (
                logits.softmax(-1) * 
                (
                    logits.log_softmax(-1) - 
                    orig_logits.log_softmax(-1)
                )
            ).sum(-1).mean()

        return loss, logits

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.opt.zero_grad()

        loss, logits = self.compute_base_loss(self.model, inputs, labels)
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        loss.backward()
        self.opt.step()

        print('Train step {}\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                self.global_iter, loss, acc1, acc5))

        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/train': loss,
                'acc/top1_train': acc1,
                'acc/top5_train': acc5,
            })
        
        return loss.item(), acc1, acc5

    def val_step(self):
        self.model.eval()

        val_subset = Subset(self.val_set, np.random.randint(0, len(self.val_set), 100))
        val_loader = DataLoader(
            val_subset,
            batch_size=100,
            shuffle=True,
            # num_workers=2,
            pin_memory=True
        )

        with torch.no_grad():
            inputs, labels = next(iter(val_loader))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            loss, logits = self.compute_base_loss(self.model, inputs, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        print('Epoch {} Validation\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                self.epoch, loss, acc1, acc5))
        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/val_avg': loss,
                'acc/top1_val_avg': acc1,
                'acc/top5_val_avg': acc5,
            })
        
        return loss.item(), acc1, acc5

    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        self.scheduler = None

        self.model.train()
        self.model.to(self.device)
        self.global_iter = 0
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.config.bs,
            shuffle=True,
            # num_workers=2,
            pin_memory=True
        )

        print("Starting finetune training")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            losses = utils.AverageMeter('Loss', ':.4e')
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            for inputs, labels in train_loader:
                self.verbose_echo("Loaded training batch")

                loss, acc1, acc5 = self.train_step(inputs, labels)

                losses.update(loss, inputs.shape[0])
                top1.update(acc1, inputs.shape[0])
                top5.update(acc5, inputs.shape[0])

                if self.global_iter % 100 == 0:
                    self.val_step()

                self.saveState(self.model, self.global_iter, self.config.model)
                self.global_iter += 1

            print('Epoch {} Train\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                self.epoch, losses.avg, top1.avg, top5.avg))

            if not self.config.debug:
                wandb.log({
                    'step': self.global_iter,
                    'loss/train_avg': losses.avg,
                    'acc/top1_train_avg': top1.avg,
                    'acc/top5_train_avg': top5.avg,
                })

            if self.scheduler:
                self.scheduler.step()

        self.saveState(self.model, self.global_iter, final=True)


class EditTrainer(BaseTrainer):
    def __init__(self, config, train_set, val_set, model_path=None):
        super().__init__(config, train_set, val_set)
        if self.config.model == 'resnet18':
            utils.prep_resnet_for_maml(self.model, layers=[3])
        elif self.config.model == 'densenet169':
            utils.prep_densenet_for_maml(self.model)

        # TODO make compatible with DenseNet
        if config.split_params:
            basic_block_predicate = lambda m: isinstance(m, torchvision.models.resnet.BasicBlock)
            n_hidden = lambda m: m.conv2.weight.shape[0]
            ConditionalLinearWrapper.wrap_model(
                self.model,
                n_hidden,
                dim=-3,  # (in_channels, out_channels, H, W)
                predicate=basic_block_predicate,
                ortho=config.ortho
            )

    def train_step(self, inputs, labels):
        self.verbose_echo("Train step")

        base_inputs, loc_inputs = torch.split(inputs, [(inputs.shape[0] + 1) // 2, inputs.shape[0] // 2])
        base_labels, loc_labels = torch.split(labels, [(labels.shape[0] + 1) // 2, labels.shape[0] // 2])

        base_inputs, base_labels = base_inputs.to(self.device), base_labels.to(self.device)
        loc_inputs, loc_labels = loc_inputs.to(self.device), loc_labels.to(self.device)

        self.model.train()
        self.opt.zero_grad()
        if self.config.learnable_lr:
            self.lr_opt.zero_grad()

        sum_l_edit = 0.0
        sum_l_loc = 0.0

        # Cache the current params and grads since we're going to modify the model during
        #  the edit process
        p_cache = {}
        for n, p in self.model.named_parameters():
            p_cache[n] = p.data.detach().clone()

        self.model.train()  # Use running statistics for edits and locality loss
        backprop_idx = random.randrange(self.config.n_edits)
        for edit_example_idx in range(self.config.n_edits):
            edit_inputs, edit_labels = next(self.train_edit_gen)
            edit_inputs, edit_labels = edit_inputs.to(self.device), edit_labels.to(self.device)

            model_edited, l_edit, lp_hist, prob_change, edit_success, inner_grad_norms = performEdits(
                self.model,
                edit_inputs,
                edit_labels,
                n_edit_steps=self.config.n_edit_steps,
                lrs=self.lrs,
                default_lr=self.config.inner_lr,
                mode="train",
                split_params=self.config.split_params
            )

            if edit_example_idx == backprop_idx:
                sum_l_edit += l_edit.item()
                self.verbose_echo(f"Computed edit loss: {l_edit}")

                with torch.no_grad():
                    loc_logits = self.model(loc_inputs)
                model_edited.train()
                edited_loc_logits = model_edited(loc_inputs)

                l_loc = (
                    loc_logits.softmax(-1).detach() * (
                        loc_logits.log_softmax(-1).detach() -
                        edited_loc_logits.log_softmax(-1)
                    )
                ).sum(-1).mean()
                sum_l_loc += l_loc.item()
                self.verbose_echo(f"Computed locality loss: {l_loc}")

                total_edit_loss = (
                    self.config.cloc * l_loc  + 
                    self.config.cedit * l_edit
                )

                if not self.config.split_params or edit_example_idx == 0:
                    total_edit_loss.backward()
                else:
                    # Only train phi/lrs using edit loss, not theta
                    edit_params = self.model.phi() + self.lrs
                    for p, g in zip(edit_params, A.grad(total_edit_loss, edit_params)):
                        if p.grad is not None:
                            p.grad += g
                        else:
                            p.grad = g.clone()
            
            for fp, p in zip(model_edited.parameters(), self.model.parameters()):
                p.data = fp.data.detach()

            # It only makes sense to train more than one edit if we've split the params
            if not self.config.split_params:
                break

        # restore pre-edit parameters without overwriting the gradients we just computed
        for n, p in self.model.named_parameters():
            p.data = p_cache[n]

        # Compute base loss
        self.model.train()  # Use batch stats for base loss
        l_base, base_logits = self.compute_base_loss(self.model, base_inputs, base_labels)
        l_base *= self.config.cbase
        l_base.backward()
        self.verbose_echo(f"Computed base loss: {l_base}")

        acc1_pre, acc5_pre = accuracy(base_logits, base_labels, topk=(1,5))
        self.verbose_echo(f"Computed pre-edit accuracies: {acc1_pre}")

        edited_base_logits = model_edited(base_inputs)
        acc1_post, acc5_post = accuracy(edited_base_logits, base_labels, topk=(1,5))
        self.verbose_echo(f"Computed post-edit accuracies: {acc1_post}")

        total_loss = (l_base + self.config.cloc * sum_l_loc + self.config.cedit * sum_l_edit).item()

        info_dict = {
            'step': self.global_iter,
            'loss/base': l_base.item(),
            'loss/edit': sum_l_edit,
            'loss/loc': sum_l_loc,
            'loss/train': total_loss,
            'edit_success_train': edit_success,
            'acc/top1_pre_train': acc1_pre,
            'acc/top5_pre_train': acc5_pre,
            'acc/top1_post_train': acc1_post,
            'acc/top5_post_train': acc5_post,
            'grad/inner': np.sum(inner_grad_norms),
            **{f'lr/lr{i}': lr.data.item() for i, lr in enumerate(self.lrs)}
        }

        if self.config.split_params:
            info_dict['grad/phi'] = nn.utils.clip_grad_norm_(self.model.phi(), 50).item()
            info_dict['grad/theta'] = nn.utils.clip_grad_norm_(self.model.theta(), 50).item()
            info_dict['norm_phi'] = torch.cat([p.flatten() for p in self.model.phi()]).norm().detach()
        else:
            info_dict['grad/all'] = nn.utils.clip_grad_norm_(self.model.parameters(), 50).item()
        if self.config.lr_grad_clip:
            info_dict['grad/lrs'] = nn.utils.clip_grad_norm_(self.lrs, self.config.lr_grad_clip).item()
        else:
            info_dict['grad/lrs'] = torch.cat([lr.grad.flatten() for lr in self.lrs]).norm().detach()

        print('Train step {}\tLoss: {:.2f}\tEdit success: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.global_iter, total_loss, edit_success, acc1_pre, acc1_post, acc5_pre, acc5_post))

        if not self.config.debug:
            wandb.log(info_dict)

        self.opt.step()
        if self.config.learnable_lr:
            self.lr_opt.step()

        return total_loss, acc1_post, acc5_post

    def val_step(self):
        self.verbose_echo("Validation step")

        top1_pre, top5_pre = [], []
        top1_post, top5_post = [], []
        losses, prob_changes, edit_successes = [], [], []

        n_val_edit_examples = len(self.val_set) // 10
        val_set, val_edit_data = random_split(self.val_set, [len(self.val_set) - n_val_edit_examples, n_val_edit_examples])
        self.val_edit_gen = editGenerator(val_edit_data, self.config.edit_bs)

        val_loader = DataLoader(
            val_set,
            batch_size=2*self.config.bs,
            shuffle=True,
            # num_workers=2,
            pin_memory=True
        )
        self.verbose_echo("Retrieved validation dataloader")

        with torch.no_grad():
            edit_num = 0
            for inputs, labels in val_loader:
                self.verbose_echo(f"Validation\tEdit number {edit_num}")

                base_inputs, loc_inputs = torch.split(inputs, [(inputs.shape[0] + 1) // 2, inputs.shape[0] // 2])
                base_labels, loc_labels = torch.split(labels, [(labels.shape[0] + 1) // 2, labels.shape[0] // 2])
                edit_inputs, edit_labels = next(self.val_edit_gen)
                self.verbose_echo("Generated edit example")

                base_inputs, base_labels = base_inputs.to(self.device), base_labels.to(self.device)
                loc_inputs, loc_labels = loc_inputs.to(self.device), loc_labels.to(self.device)
                edit_inputs, edit_labels = edit_inputs.to(self.device), edit_labels.to(self.device)

                self.model.train()
                l_base, base_logits = self.compute_base_loss(self.model, base_inputs, base_labels)
                self.verbose_echo(f"Computed base loss: {l_base}")

                acc1_pre, acc5_pre = accuracy(base_logits, base_labels, topk=(1, 5))
                self.verbose_echo(f"Computed pre-edit accuracies: {acc1_pre}")

                top1_pre.append(acc1_pre)
                top5_pre.append(acc5_pre)

                model_edited, l_edit, lp_hist, prob_change, edit_success, _ = performEdits(
                    self.model,
                    edit_inputs,
                    edit_labels,
                    n_edit_steps=self.config.n_edit_steps,
                    lrs=self.lrs,
                    default_lr=self.config.inner_lr,
                    mode="val",
                    split_params=self.config.split_params
                )
                prob_changes.append(prob_change)
                edit_successes.append(edit_success)

                self.verbose_echo(f"Computed edit loss: {l_edit}")

                model_edited.train()
                loc_logits = self.model(loc_inputs)
                edited_loc_logits = model_edited(loc_inputs)
                l_loc = (
                    loc_logits.softmax(-1) * 
                    (
                        loc_logits.log_softmax(-1) - 
                        edited_loc_logits.log_softmax(-1)
                    )
                ).sum(-1).mean()

                self.verbose_echo(f"Computed locality loss: {l_loc}")

                total_edit_loss = (
                    self.config.cloc * l_loc  + 
                    self.config.cedit * l_edit
                )
                total_loss = l_base.item() + total_edit_loss.item()
                losses.append(total_loss)

                edited_base_logits = model_edited(base_inputs)
                acc1_post, acc5_post = accuracy(edited_base_logits, base_labels, topk=(1, 5))

                self.verbose_echo(f"Computed post-edit accuracies: {acc1_post}")

                top1_post.append(acc1_post)
                top5_post.append(acc5_post)

                edit_num += 1
                if edit_num >= 10:  # validate on 10 edits
                    break

        print('Epoch {} Validation\tLoss: {:.2f}\tEdit success: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.epoch, np.mean(losses), np.mean(edit_successes), np.mean(top1_pre),
                np.mean(top1_post), np.mean(top5_pre), np.mean(top5_post)))
        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/val': np.mean(losses),
                'edit_success_val': np.mean(edit_successes),
                'acc/top1_pre_val': np.mean(top1_pre),
                'acc/top1_post_val': np.mean(top1_post),
                'acc/top5_pre_val': np.mean(top5_pre),
                'acc/top5_post_val': np.mean(top5_post),
            })

    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        if self.config.weight_decay:
            if self.config.split_params:
                self.opt = torch.optim.Adam([
                    {'params': self.model.theta()},
                    {'params': self.model.phi(),'weight_decay': self.config.weight_decay}
                ], lr=self.config.outer_lr)
            else:
                self.opt = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config.outer_lr,
                    weight_decay=self.config.weight_decay
                )
        else:
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.outer_lr
            )
        self.scheduler = None
        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.model.inner_params()
        ]
        self.lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)
        self.lr_scheduler = None
        '''
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.lr_opt,
            T_max=self.config.epochs
        )
        '''

        self.model.train()
        self.model.to(self.device)
        self.global_iter = 0

        self.verbose_echo("Starting editable training")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            losses = utils.AverageMeter('Loss', ':.4e')
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            self.verbose_echo(f"Epoch {epoch}")

            n_train_edit_examples = len(self.train_set) // 10
            train_set, train_edit_data = random_split(self.train_set, [len(self.train_set) - n_train_edit_examples, n_train_edit_examples])
            self.train_edit_gen = editGenerator(train_edit_data, self.config.edit_bs)

            train_loader = DataLoader(
                self.train_set,
                batch_size=2*self.config.bs,
                shuffle=True,
                # num_workers=2,
                pin_memory=True
            )

            for inputs, labels in train_loader:
                self.verbose_echo("Loaded training batch")

                loss, acc1, acc5 = self.train_step(inputs, labels)

                losses.update(loss, inputs.shape[0])
                top1.update(acc1, inputs.shape[0])
                top5.update(acc5, inputs.shape[0])

                if self.global_iter % 100 == 0:
                    self.val_step()

                self.saveState(self.model, self.global_iter, self.config.model)
                if self.config.learnable_lr:
                    self.saveState(self.lrs, self.global_iter, name='lr')
                self.global_iter += 1

            print('Epoch {} Train\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                self.epoch, losses.avg, top1.avg, top5.avg))

            if not self.config.debug:
                wandb.log({
                    'step': self.global_iter,
                    'loss/train_avg': losses.avg,
                    'acc/top1_train_avg': top1.avg,
                    'acc/top5_train_avg': top5.avg,
                    'lr': self.opt.param_groups[0]['lr'],
                    'lr_lr': self.lr_opt.param_groups[0]['lr']
                })

            if self.scheduler:
                self.scheduler.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

        self.saveState(self.model, self.global_iter, final=True)
        if self.config.learnable_lr:
            self.saveState(self.lrs, self.global_iter, final=True, name='lr')


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    loc = utils.sailPreprocess()
    train_set = loadCIFAR(loc, 'train') if config.dataset == 'cifar10' else loadImageNet(loc, 'train')
    val_set = loadCIFAR(loc, 'val') if config.dataset == 'cifar10' else loadImageNet(loc, 'val')

    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.loc = loc

    if config.task == 'editable':
        trainer = EditTrainer(config, train_set, val_set)

    else:
        trainer = BaseTrainer(config, train_set, val_set)

    trainer.run()


if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    main()
