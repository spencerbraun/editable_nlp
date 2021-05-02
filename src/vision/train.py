import os
import argparse
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

import wandb
import utils
from data_process import loadCIFAR, loadImageNet
from config import TrainConfig, EditConfig
from evaluate import accuracy, get_logprobs

eps = np.finfo(np.float32).eps.item()
model_paths = {
    ('cifar', 'resnet18'): 'models/cifar/resnet18/finetune/finetune_epoch199_ts9999.20210429.19.04.1619750982',
    ('imagenet', 'resnet18'): None,
    ('imagenet', 'densenet169'): None,
}

class BaseTrainer:
    def __init__(self, config, train_set, val_set, model_path=None):

        # config
        self.config = config
        self.model_dir = (
            f'{self.config.write_loc}/models/{self.config.dataset}/{self.config.arch}/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
        )
        self.num_classes = 10 if self.config.dataset == 'cifar' else 1000
        load_model = densenet169 if self.config.arch == 'densenet169' else resnet18
        pretrained = self.config.dataset == 'imagenet'  # torchvision models are pretrained on ImageNet
        self.model = (
            utils.loadOTSModel(load_model, self.num_classes, pretrained=pretrained) if not model_path else
            utils.loadTrainedModel(model_path, load_model, self.num_classes)
        )

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not self.config.debug:
            wandb.init(
                project='patchable-cnn',
                entity='patchable-lm',
                config=self.config,
                name=f"{self.config.dataset}/{self.config.arch}/{self.config.task}_{self.timestamp}",
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

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.model.train()
        self.opt.zero_grad()

        logits = self.model(inputs)
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        loss = F.cross_entropy(logits, labels)
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
        
        return loss, acc1, acc5

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

            logits = self.model(inputs)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            loss = F.cross_entropy(logits, labels)

        print('Epoch {} Validation\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                self.epoch, loss, acc1, acc5))
        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/val_avg': loss,
                'acc/top1_val_avg': acc1,
                'acc/top5_val_avg': acc5,
            })

    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.outer_lr)
        self.scheduler = None
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.config.outer_lr, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=200)

        self.model.train()
        self.model.to(self.device)
        self.global_iter = 0
        train_loader = DataLoader(self.train_set, batch_size=config.bs, shuffle=True, num_workers=2)

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

                self.saveState(self.model, train_step, self.config.arch)
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


class EditTrainer(BaseTrainer):
    def __init__(self, config, train_set, val_set, model_path=None):
        super().__init__(config, train_set, val_set, model_path)
        self.train_edit_gen = self.editGenerator('train')
        self.val_edit_gen = self.editGenerator('val')

    def getEditParams(self, model):
        if self.config.arch == 'resnet18':
            return model.layer3.parameters()
        else:
            return model.features.denseblock3.parameters()

    def editGenerator(self, ds='train'):
        edit_dataset = self.train_set if ds == 'train' else self.val_set
        sampler = RandomSampler(edit_dataset, replacement=True, num_samples=1)
        while True:
            for inputs, labels in DataLoader(edit_dataset, sampler=sampler, batch_size=1, num_workers=2):
                inputs = inputs.to(self.device)
                edit_labels = torch.randint_like(labels, self.num_classes, device=self.device)
                yield inputs, edit_labels

    def performEdits(self, edit_inputs, edit_labels):
        model_ = copy.deepcopy(self.model)
        loss_hist, lp_hist, ll_change_hist = [], [], []

        model_.eval()
        lp_hist.append(get_logprobs(model_, edit_inputs, edit_labels))

        model_.train()
        param_groups = [
            {'params': p, 'lr': None} 
            for p in self.getEditParams(model_)
        ]
        inner_opt = torch.optim.SGD(param_groups)

        with higher.innerloop_ctx(
            model_,
            inner_opt,
            override={'lr': self.lrs} if self.config.learnable_lr else None,
            copy_initial_weights=False, 
            track_higher_grads=True
        ) as (fmodel, diffopt):
            for edit_step in range(self.config.n_edit_steps):
                fmodel.eval()  # needed for batchnorm to work properly on batch size of 1
                loss = F.cross_entropy(fmodel(edit_inputs), edit_labels)
                diffopt.step(loss)

                lp = get_logprobs(fmodel, edit_inputs, edit_labels)
                ll_change = torch.mean(abs(lp_hist[0]) - abs(lp)) / (abs(lp_hist[0]) + eps)
                print(f"log prob history: {lp_hist}")
                print(f"Edit step {edit_step}; ll change {ll_change}, log prob {lp}, loss {loss}")

                loss_hist.append(loss)
                lp_hist.append(lp)
                ll_change_hist.append(ll_change)

            model_.load_state_dict(fmodel.state_dict())

        return model_, loss_hist, lp_hist, ll_change_hist

    def train_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        edit_inputs, edit_labels = next(self.train_edit_gen)

        self.model.train()
        self.opt.zero_grad()
        if self.config.learnable_lr:
            self.lr_opt.zero_grad()

        total_loss = 0.0

        base_logits = self.model(inputs)
        acc1_pre, acc5_pre = accuracy(base_logits, labels, topk=(1,5))
        l_base = F.cross_entropy(base_logits, labels)  # TODO make loss a config param
        l_base.backward()
        total_loss += l_base

        param_groups = [
            {'params': p, 'lr': None} 
            for p in self.getEditParams(self.model)
        ]
        inner_opt = (
            torch.optim.SGD(param_groups) if self.config.learnable_lr
            else torch.optim.SGD(
                self.getEditParams(self.model),
                lr=self.config.inner_lr
            )
        )

        with higher.innerloop_ctx(
            self.model,
            inner_opt,
            override={'lr': self.lrs} if self.config.learnable_lr else None,
            copy_initial_weights=False,
            track_higher_grads=True,
        ) as (fmodel, diffopt):
            for edit_step in range(self.config.n_edit_steps):
                fmodel.eval()  # needed for batchnorm to work properly on batch size of 1
                loss = F.cross_entropy(fmodel(edit_inputs), edit_labels)
                diffopt.step(loss)

            edit_logits = fmodel(edit_inputs)
            l_edit = F.cross_entropy(edit_logits, edit_labels)

            edited_base_logits = fmodel(inputs)
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
            total_loss += total_edit_loss

        self.opt.step()
        if self.config.learnable_lr:
            self.lr_opt.step()
            self.saveState(self.lrs, self.global_iter, name='lr')

        print('Train step {}\tLoss: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.global_iter, total_loss, acc1_pre, acc1_post, acc5_pre, acc5_post))

        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/base': l_base,
                'loss/edit': l_edit,
                'loss/local': l_loc,
                'loss/train': total_loss,
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
        losses, ll_changes = [], []

        val_subset = Subset(self.val_set, np.random.randint(0, len(self.val_set), 100))
        val_loader = DataLoader(
            val_subset,
            batch_size=100,
            shuffle=True,
            num_workers=2
        )

        for edit_num, (edit_inputs, edit_labels) in enumerate(list(itertools.islice(self.val_edit_gen, 10))):
            top1 = utils.AverageMeter('Acc@1', ':6.2f')
            top5 = utils.AverageMeter('Acc@5', ':6.2f')

            total_loss = 0.0

            with torch.no_grad():
                self.model.eval()
                inputs, labels = next(iter(val_loader))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                base_logits = self.model(inputs)
                acc1, acc5 = accuracy(base_logits, labels, topk=(1, 5))
                l_base = F.cross_entropy(base_logits, labels)
                total_loss += l_base

                top1.update(acc1, inputs.shape[0])
                top5.update(acc5, inputs.shape[0])

            top1_pre.append(top1.avg)
            top5_pre.append(top5.avg)

            top1.reset()
            top5.reset()

            model_edited, loss_hist, lp_hist, ll_change_hist = self.performEdits(edit_inputs, edit_labels)
            l_edit = loss_hist[-1]
            ll_change = torch.mean(abs(lp_hist[0]) - abs(lp_hist[-1])) / (abs(lp_hist[0]) + eps)
            ll_changes.append(ll_change)

            with torch.no_grad():
                model_edited.eval()

                edited_base_logits = model_edited(inputs)
                acc1, acc5 = accuracy(edited_base_logits, labels, topk=(1, 5))
                l_loc = (
                    F.softmax(base_logits, dim=-1) *
                    (
                        F.log_softmax(base_logits, dim=-1) - 
                        F.log_softmax(edited_base_logits, dim=-1)
                    )).sum(-1).mean()

                top1.update(acc1, inputs.shape[0])
                top5.update(acc5, inputs.shape[0])

                total_edit_loss = (
                    self.config.cloc * l_loc  + 
                    self.config.cedit * l_edit
                ) # / self.config.n_edits
                total_loss += total_edit_loss

            losses.append(total_loss.item())
            top1_post.append(top1.avg)
            top5_post.append(top5.avg)

        print('Epoch {} Validation\tLoss: {:.2f}\tPre-edit top-1 acc: {:.2f}\tPost-edit top-1 acc: {:.2f}\t' \
                'Pre-edit top-5 acc: {:.2f}\tPost-edit top-5 acc: {:.2f}'.format(
                self.epoch, np.mean(losses), np.mean(top1_pre), np.mean(top1_post),
                np.mean(top5_pre), np.mean(top5_post)))
        if not self.config.debug:
            wandb.log({
                'step': self.global_iter,
                'loss/val': np.mean(losses),
                'acc/top1_pre_val': np.mean(top1_pre),
                'acc/top1_post_val': np.mean(top1_post),
                'acc/top5_pre_val': np.mean(top5_pre),
                'acc/top5_post_val': np.mean(top5_post),
            })

    def run(self):
        self.lrs = [
            torch.nn.Parameter(torch.tensor(self.config.inner_lr)) 
            for p in self.getEditParams(self.model)
        ]
        self.lr_opt = torch.optim.Adam(self.lrs, lr=self.config.lr_lr)

        super().run()

        if self.config.learnable_lr:
            self.saveState(self.lrs, self.global_iter, final=True, name='lr')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--editable', action='store_true')
    parser.add_argument('--dataset',
        choices=['cifar', 'imagenet'],
        default='imagenet',
        help='Which dataset to use.',
    )
    parser.add_argument('--arch',
        choices=['resnet18', 'densenet169'],
        default='resnet18',
        help='Which model architecture to use.',
    )
    parser.add_argument('--bs', type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loc = utils.sailPreprocess()

    train_set = loadCIFAR(loc, 'train') if args.dataset == 'cifar' else loadImageNet(loc, 'train')
    val_set = loadCIFAR(loc, 'val') if args.dataset == 'cifar' else loadImageNet(loc, 'val')

    if args.editable:
        config = EditConfig()
        config.write_loc = loc
        config.dataset = args.dataset
        config.arch = args.arch
        config.bs = args.bs
        model_path = model_paths[(config.dataset, config.arch)]

        trainer = EditTrainer(config, train_set, val_set, model_path)

    else:
        config = TrainConfig()
        config.write_loc = loc
        config.dataset = args.dataset
        config.arch = args.arch
        config.bs = args.bs

        trainer = BaseTrainer(config, train_set, val_set)

    trainer.run()
