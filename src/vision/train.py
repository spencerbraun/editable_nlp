import os
import glob
import time
import random
import copy
from datetime import datetime
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
from torchvision.models import resnet18, densenet169
import higher

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

import wandb
import utils
from data_process import loadCIFAR, loadImageNet
from evaluate import accuracy, get_logprobs, editGenerator, performEdits

eps = np.finfo(np.float32).eps.item()


class BaseTrainer:
    def __init__(self, config, train_set, val_set):

        # config
        self.config = config

        self.model_dir = (
            f'{self.config.write_loc}/models/{self.config.dataset}/{self.config.model}/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
        )

        num_classes = len(train_set.classes)

        load_model = densenet169 if config.model == 'densenet169' else resnet18
        self.model = utils.loadOTSModel(load_model, num_classes, self.config.pretrained)
        if not self.config.pretrained and not OmegaConf.is_missing(self.config, 'model_path'):
            model_path = os.path.join(self.config.write_loc, self.config.model_path)
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")
        self.original_model = copy.deepcopy(self.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.original_model.to(self.device)

        # outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}"
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
                dir=self.config.write_loc,
            )
            wandb.watch(self.model)

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
                f"; ".join([f"{key} {val}" for key,val in kwargs.items()])
            ))

    def configure_data(self, train_set, val_set):
        self.train_set = train_set
        self.val_set = val_set

    def compute_base_loss(self, inputs, labels):
        if self.config.loss == 'cross_entropy':
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, labels)

        elif self.config.loss == 'kl':
            logits = self.model(inputs)
            self.original_model.eval()
            with torch.no_grad():
                orig_logits = self.original_model(inputs)

            loss = (
                F.softmax(logits, dim=-1) *
                (
                    F.log_softmax(logits, dim=-1) - 
                    F.log_softmax(orig_logits, dim=-1)
                )).sum(-1).mean()

        return loss, logits

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.opt.zero_grad()

        loss, logits = self.compute_base_loss(inputs, labels)
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
            num_workers=2
        )

        with torch.no_grad():
            inputs, labels = next(iter(val_loader))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            loss, logits = self.compute_base_loss(inputs, labels)
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
        self.global_iter = 0
        train_loader = DataLoader(self.train_set, batch_size=self.config.bs, shuffle=True, num_workers=2)

        print("Starting training")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            losses = utils.AverageMeter('Loss', ':.4e')
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            for train_step, (inputs, labels) in enumerate(train_loader):

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
                    'lr': self.opt.param_groups[0]['lr'],
                })

            if self.scheduler:
                self.scheduler.step()

        self.saveState(self.model, self.global_iter, final=True)


class EditTrainer(BaseTrainer):
    def __init__(self, config, train_set, val_set, model_path=None):
        super().__init__(config, train_set, val_set)
        if self.config.model == 'resnet18':
            utils.prep_resnet_for_maml(self.model)
        elif self.config.model == 'densenet169':
            utils.prep_densenet_for_maml(self.model)

    def configure_data(self, train_set, val_set):
        n_train_edit_examples = len(train_set) // 10
        n_val_edit_examples = len(val_set) // 10
        self.train_set, train_edit_data = random_split(train_set, [len(train_set) - n_train_edit_examples, n_train_edit_examples])
        self.val_set, val_edit_data = random_split(val_set, [len(val_set) - n_val_edit_examples, n_val_edit_examples])

        self.train_edit_gen = editGenerator(train_edit_data, self.config.n_edits)
        self.val_edit_gen = editGenerator(val_edit_data, self.config.n_edits)

    def train_step(self, inputs, labels):
        base_inputs, loc_inputs = torch.split(inputs, [(inputs.shape[0] + 1) // 2, inputs.shape[0] // 2])
        base_labels, loc_labels = torch.split(labels, [(labels.shape[0] + 1) // 2, labels.shape[0] // 2])
        edit_inputs, edit_labels = next(self.train_edit_gen)

        base_inputs, base_labels = base_inputs.to(self.device), base_labels.to(self.device)
        loc_inputs, loc_labels = loc_inputs.to(self.device), loc_labels.to(self.device)
        edit_inputs, edit_labels = edit_inputs.to(self.device), edit_labels.to(self.device)

        self.model.train()
        self.opt.zero_grad()
        if self.config.learnable_lr:
            self.lr_opt.zero_grad()

        total_loss = 0.0

        l_base, base_logits = self.compute_base_loss(base_inputs, base_labels)
        acc1_pre, acc5_pre = accuracy(base_logits, base_labels, topk=(1,5))
        l_base.backward()
        total_loss += l_base.item()

        model_edited, l_edit, lp_hist, ll_change, edit_success = performEdits(
            self.model,
            edit_inputs,
            edit_labels,
            n_edit_steps=self.config.n_edit_steps,
            lrs=self.lrs,
            default_lr=self.config.inner_lr,
            mode="train"
        )

        loc_logits = self.model(loc_inputs)
        edited_loc_logits = model_edited(loc_inputs)
        l_loc = (
            F.softmax(loc_logits.detach(), dim=-1) *
            (
                F.log_softmax(loc_logits.detach(), dim=-1) - 
                F.log_softmax(edited_loc_logits, dim=-1)
            )).sum(-1).mean()

        total_edit_loss = (
            self.config.cloc * l_loc  + 
            self.config.cedit * l_edit
        )
        total_edit_loss.backward()
        total_loss += total_edit_loss.item()

        edited_base_logits = model_edited(base_inputs)
        acc1_post, acc5_post = accuracy(edited_base_logits, base_labels, topk=(1,5))

        self.opt.step()
        if self.config.learnable_lr:
            self.lr_opt.step()

        print('Train step {}\tLoss: {:.2f}\tEdit success: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.global_iter, total_loss, edit_success, acc1_pre, acc1_post, acc5_pre, acc5_post))

        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/base': l_base,
                'loss/edit': l_edit,
                'loss/loc': l_loc,
                'loss/train': total_loss,
                'edit_success_train': edit_success,
                'acc/top1_pre_train': acc1_pre,
                'acc/top5_pre_train': acc5_pre,
                'acc/top1_post_train': acc1_post,
                'acc/top5_post_train': acc5_post,
                **{f'lr/lr{i}': lr.data.item() for i, lr in enumerate(self.lrs)},
            })

        return total_loss, acc1_post, acc5_post

    def val_step(self):
        top1_pre, top5_pre = [], []
        top1_post, top5_post = [], []
        losses, ll_changes, edit_successes = [], [], []

        val_loader = DataLoader(self.val_set, batch_size=2*self.config.bs, shuffle=True, num_workers=2)

        with torch.no_grad():
            for edit_num, (inputs, labels) in enumerate(val_loader):
                if edit_num >= 10:  # validate on 10 edits
                    break

                base_inputs, loc_inputs = torch.split(inputs, [(inputs.shape[0] + 1) // 2, inputs.shape[0] // 2])
                base_labels, loc_labels = torch.split(labels, [(labels.shape[0] + 1) // 2, labels.shape[0] // 2])
                edit_inputs, edit_labels = next(self.val_edit_gen)

                base_inputs, base_labels = base_inputs.to(self.device), base_labels.to(self.device)
                loc_inputs, loc_labels = loc_inputs.to(self.device), loc_labels.to(self.device)
                edit_inputs, edit_labels = edit_inputs.to(self.device), edit_labels.to(self.device)

                self.model.eval()
                l_base, base_logits = self.compute_base_loss(base_inputs, base_labels)
                acc1_pre, acc5_pre = accuracy(base_logits, base_labels, topk=(1, 5))

                top1_pre.append(acc1_pre)
                top5_pre.append(acc5_pre)

                model_edited, l_edit, lp_hist, ll_change, edit_success = performEdits(
                    self.model,
                    edit_inputs,
                    edit_labels,
                    n_edit_steps=self.config.n_edit_steps,
                    lrs=self.lrs,
                    default_lr=self.config.inner_lr,
                    mode="val"
                )
                ll_changes.append(ll_change)
                edit_successes.append(edit_success)

                model_edited.eval()
                loc_logits = self.model(loc_inputs)
                edited_loc_logits = model_edited(loc_inputs)
                l_loc = (
                    F.softmax(loc_logits, dim=-1) *
                    (
                        F.log_softmax(loc_logits, dim=-1) - 
                        F.log_softmax(edited_loc_logits, dim=-1)
                    )).sum(-1).mean()

                total_edit_loss = (
                    self.config.cloc * l_loc  + 
                    self.config.cedit * l_edit
                )
                total_loss = l_base.item() + total_edit_loss.item()
                losses.append(total_loss)

                edited_base_logits = model_edited(base_inputs)
                acc1_post, acc5_post = accuracy(edited_base_logits, base_labels, topk=(1, 5))

                top1_post.append(acc1_post)
                top5_post.append(acc5_post)

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

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        self.scheduler = None
        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.model.inner_params()
        ]
        self.lr_opt = torch.optim.SGD(self.lrs, lr=self.config.lr_lr)

        self.model.train()
        self.global_iter = 0
        train_loader = DataLoader(self.train_set, batch_size=2*self.config.bs, shuffle=True, num_workers=2)

        print("Starting training")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            losses = utils.AverageMeter('Loss', ':.4e')
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            for train_step, (inputs, labels) in enumerate(train_loader):

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
                })

            if self.scheduler:
                self.scheduler.step()

        self.saveState(self.model, self.global_iter, final=True)
        if self.config.learnable_lr:
            self.saveState(self.lrs, self.global_iter, final=True, name='lr')


@hydra.main(config_path='config/train', config_name='config')
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    loc = utils.sailPreprocess()

    train_set = loadCIFAR(loc, 'train') if config.dataset == 'cifar10' else loadImageNet(loc, 'train')
    val_set = loadCIFAR(loc, 'val') if config.dataset == 'cifar10' else loadImageNet(loc, 'val')

    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.write_loc = loc

    if config.task == 'editable':
        trainer = EditTrainer(config, train_set, val_set)

    else:
        trainer = BaseTrainer(config, train_set, val_set)

    trainer.run()


if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    main()
