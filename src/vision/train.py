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
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.models import resnet18, densenet169
import higher

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

import wandb
import utils
from data_process import loadCIFAR, loadImageNet
from evaluate import accuracy, get_logprobs

eps = np.finfo(np.float32).eps.item()


class BaseTrainer:
    def __init__(self, config, train_set, val_set):

        # config
        self.config = config

        self.model_dir = (
            f'{self.config.write_loc}/models/{self.config.dataset}/{self.config.model}/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
        )

        load_model = densenet169 if config.model == 'densenet169' else resnet18
        self.model = utils.loadOTSModel(load_model, self.config.num_classes, self.config.pretrained)
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

        self.train_set = train_set
        self.val_set = val_set

        self.configure_optimizers()

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

    def getEditParams(self, model):
        return model.parameters()

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

                self.saveState(self.model, train_step, self.config.model)
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

        self.saveState(self.model, 0, final=True)
        if self.config.learnable_lr:
            self.saveState(self.lrs, self.global_iter, final=True, name='lr')
    
    def configure_optimizers(self):
        self.opt = torch.optim.Adam(self.getEditParams(self.model), lr=self.config.outer_lr)
        self.scheduler = None
        self.lrs = None
        self.lr_opt = None


class EditTrainer(BaseTrainer):
    def __init__(self, config, train_set, val_set, model_path=None):
        super().__init__(config, train_set, val_set)
        self.train_edit_gen = self.editGenerator('train')
        self.val_edit_gen = self.editGenerator('val')

    def getEditParams(self, model):
        if self.config.model == 'resnet18':
            return model.layer3.parameters()
        else:
            return model.features.denseblock3.parameters()

    def editGenerator(self, ds='train'):
        edit_dataset = self.train_set if ds == 'train' else self.val_set
        sampler = RandomSampler(edit_dataset, replacement=True, num_samples=1)
        while True:
            for inputs, labels in DataLoader(edit_dataset, sampler=sampler, batch_size=1, num_workers=2):
                inputs = inputs.to(self.device)
                edit_labels = torch.randint_like(labels, self.config.num_classes, device=self.device)
                yield inputs, edit_labels

    def performEdits(self, model, edit_inputs, edit_labels):
        lp_hist = []
        l_edit, ll_change, edit_success = 0.0, 0.0, 0.0

        model.eval()
        lp_hist.append(get_logprobs(model, edit_inputs, edit_labels))

        param_groups = [
            {'params': p, 'lr': None} 
            for p in self.getEditParams(model)
        ]
        inner_opt = (
            torch.optim.SGD(param_groups) if self.config.learnable_lr
            else torch.optim.SGD(
                self.getEditParams(model),
                lr=self.config.inner_lr
            )
        )

        with torch.enable_grad():
            with higher.innerloop_ctx(
                model,
                inner_opt,
                override={'lr': self.lrs} if self.config.learnable_lr else None,
                copy_initial_weights=False,
                track_higher_grads=True
            ) as (fmodel, diffopt):
                for edit_step in range(self.config.n_edit_steps):
                    fmodel.eval()  # needed for batchnorm to work properly on batch size of 1
                    edit_logits = fmodel(edit_inputs)
                    loss = F.cross_entropy(edit_logits, edit_labels)
                    diffopt.step(loss)

                    lp = get_logprobs(fmodel, edit_inputs, edit_labels)
                    lp_hist.append(lp)

        edit_logits = fmodel(edit_inputs)
        l_edit = F.cross_entropy(edit_logits, edit_labels)
        edit_success = np.mean(torch.eq(torch.argmax(edit_logits, -1), edit_labels).cpu().numpy())
        ll_change = np.mean(abs(lp_hist[0]) - abs(lp_hist[-1]) / (abs(lp_hist[0]) + eps))

        print("Log prob history:", ["{:.2f}".format(i) for i in lp_hist])
        print("Edit step {}\tll change {:.2f}\tLog prob {:.2f}\tLoss {:.2f}".format(
                edit_step, ll_change, lp_hist[-1], l_edit))

        return fmodel, l_edit, lp_hist, ll_change, edit_success

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        edit_inputs, edit_labels = next(self.train_edit_gen)

        self.model.train()
        self.opt.zero_grad()
        if self.config.learnable_lr:
            self.lr_opt.zero_grad()

        total_loss = 0.0

        l_base, base_logits = self.compute_base_loss(inputs, labels)
        acc1_pre, acc5_pre = accuracy(base_logits, labels, topk=(1,5))
        l_base.backward()
        total_loss += l_base.item()

        model_edited, l_edit, lp_hist, ll_change, edit_success = self.performEdits(
            self.model,
            edit_inputs,
            edit_labels
        )

        edited_base_logits = model_edited(inputs)
        acc1_post, acc5_post = accuracy(edited_base_logits, labels, topk=(1,5))
        l_loc = (
            F.softmax(base_logits.detach(), dim=-1) *
            (
                F.log_softmax(base_logits.detach(), dim=-1) - 
                F.log_softmax(edited_base_logits, dim=-1)
            )).sum(-1).mean()

        total_edit_loss = (
            self.config.cloc * l_loc  + 
            self.config.cedit * l_edit
        )  # / self.config.n_edits
        total_edit_loss.backward()
        total_loss += total_edit_loss.item()

        self.opt.step()
        if self.config.learnable_lr:
            self.lr_opt.step()
            self.saveState(self.lrs, self.global_iter, name='lr')

        print('Train step {}\tLoss: {:.2f}\tEdit success: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.global_iter, total_loss, edit_success, acc1_pre, acc1_post, acc5_pre, acc5_post))

        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/base': l_base,
                'loss/edit': l_edit,
                'loss/local': l_loc,
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

        val_subset = Subset(self.val_set, np.random.randint(0, len(self.val_set), 100))
        val_loader = DataLoader(
            val_subset,
            batch_size=100,
            shuffle=True,
            num_workers=2
        )

        total_loss = 0.0

        with torch.no_grad():
            self.model.eval()
            inputs, labels = next(iter(val_loader))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            l_base, base_logits = self.compute_base_loss(inputs, labels)
            acc1_pre, acc5_pre = accuracy(base_logits, labels, topk=(1, 5))
            total_loss += l_base.item()

            top1_pre.append(acc1_pre)
            top5_pre.append(acc5_pre)

            for edit_num, (edit_inputs, edit_labels) in enumerate(list(itertools.islice(self.val_edit_gen, 10))):
                
                model_edited, l_edit, lp_hist, ll_change, edit_success = self.performEdits(
                    self.model,
                    edit_inputs,
                    edit_labels,
                )
                ll_changes.append(ll_change)
                edit_successes.append(edit_success)

                model_edited.eval()
                edited_base_logits = model_edited(inputs)
                acc1_post, acc5_post = accuracy(edited_base_logits, labels, topk=(1, 5))
                l_loc = (
                    F.softmax(base_logits, dim=-1) *
                    (
                        F.log_softmax(base_logits, dim=-1) - 
                        F.log_softmax(edited_base_logits, dim=-1)
                    )).sum(-1).mean()

                total_edit_loss = (
                    self.config.cloc * l_loc  + 
                    self.config.cedit * l_edit
                ) # / self.config.n_edits
                total_loss += total_edit_loss.item()
                losses.append(total_loss)

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

    def configure_optimizers(self):
        super().configure_optimizers()
        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.getEditParams(self.model)
        ]
        self.lr_opt = torch.optim.SGD(self.lrs, lr=self.config.lr_lr)


@hydra.main(config_path='config', config_name='config')
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
