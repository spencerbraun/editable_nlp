import os
import argparse
import glob
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, densenet169
import wandb

import utils
from data_process import loadCIFAR, loadImageNet
from config import TrainConfig


class BaseTrainer:
    def __init__(self, config, train_loader, val_loader, model_path=None):

        # config
        self.config = config
        self.model_dir = (
            f'{self.config.write_loc}/models/{self.config.dataset}/{self.config.arch}/finetune' if self.config.task == 'finetune'
            else f'{self.config.write_loc}/models'
        )
        load_model = resnet18 if self.config.arch == 'resnet18' else densenet169
        self.model = (
            utils.loadOTSModel(load_model, pretrained=False) if not model_path else
            utils.loadTrainedModel(model_path, load_model)
        )

        # outfiles
        self.timestamp = datetime.now().strftime("%Y%m%d.%H.%m.%s")
        self.hyperspath = f"{self.model_dir}/hypers.{self.timestamp}"
        self.errpath = f"{self.config.write_loc}/errors/errors_{self.timestamp}"
        self.statepath = (
            lambda model, epoch, step: 
            f"{self.model_dir}/{model}_epoch{epoch}_ts{step}.{self.timestamp}"
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
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
            elif (self.epoch % 25 == 0):
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

    def train_step(self):

        self.model.train()

        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            scores = self.model(images)
            acc1, acc5 = utils.accuracy(scores, labels, topk=(1,5))
            loss = F.cross_entropy(scores, labels)
            loss.backward()
            self.optimizer.step()

            losses.update(loss)
            top1.update(acc1)
            top5.update(acc5)

            if not self.config.debug:
                wandb.log({
                    'step': self.global_iter,
                    'loss/train': loss,
                    'acc/top1_train': acc1,
                    'acc/top5_train': acc5,
                })
            
            self.global_iter += 1
        
        self.saveState(self.model, step, self.config.arch)

        print('Epoch {} Train\tVal loss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
              self.epoch, losses.avg, top1.avg, top5.avg))

        if not self.config.debug:
            wandb.log({
                'epoch': self.epoch,
                'loss/train_avg': losses.avg,
                'acc/top1_train_avg': top1.avg,
                'acc/top5_train_avg': top5.avg,
                'lr': self.optimizer.param_groups[0]['lr'],
            })

    def val_step(self):

        self.model.eval()

        losses = utils.AverageMeter('Loss', ':.4e')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')

        with torch.no_grad():
            for val_step, (images, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                scores = self.model(images)
                acc1, acc5 = utils.accuracy(scores, labels, topk=(1, 5))
                loss = F.cross_entropy(scores, labels)

                losses.update(loss)
                top1.update(acc1)
                top5.update(acc5)

            print('Epoch {} Val\tLoss: {:.2f}\tTop-1 acc: {:.2f}\tTop-5 acc: {:.2f}'.format(
                  self.epoch, losses.avg, top1.avg, top5.avg))
            if not self.config.debug:
                wandb.log({
                    'epoch': self.epoch,
                    'loss/val_avg': losses.avg,
                    'acc/top1_val_avg': top1.avg,
                    'acc/top5_val_avg': top5.avg,
                })

    def run(self):
        if not self.config.debug:
            torch.save(self.config, self.hyperspath)
        
        self.model.to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters())
        # scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** (epoch // 5))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.global_iter = 0
        print("Starting training")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self.train_step()
            self.val_step()
            self.scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--finetune', action='store_true')
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

    torch.manual_seed(args.seed)

    loc = utils.sailPreprocess()

    train_loader = loadCIFAR(loc, 'train', args.bs) if args.dataset == 'cifar' else loadImageNet(loc, 'train', args.bs)
    val_loader = loadCIFAR(loc, 'val', args.bs) if args.dataset == 'cifar' else loadImageNet(loc, 'val', args.bs)

    if args.finetune:
        config = TrainConfig()
        config.write_loc = loc
        config.dataset = args.dataset
        config.arch = args.arch

        trainer = BaseTrainer(config, train_loader, val_loader)
        trainer.run()
