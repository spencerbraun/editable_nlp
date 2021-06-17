import logging
import os
import tempfile
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from omegaconf import DictConfig
import wandb

from experimental.build import build_optimizer, build_scheduler
import experimental.utils as utils

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


class BaseTrainer:

    def __init__(self, model, config: DictConfig, train_set: Dataset, val_set: Dataset):

        self.model = model
        self.config = config
        self.train_set = train_set
        self.val_set = val_set

        run_dir = os.getcwd()
        self.model_dir = os.path.join(run_dir, 'models')
        if not os.path.exists(self.model_dir) and not self.config.debug:
            os.makedirs(self.model_dir)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # outfiles
        self.run_id = run_dir.split('/')[-1]
        self.statepath = (
            lambda model, epoch, step: 
            f"{self.model_dir}/{model}_epoch{epoch}_ts{step}.{self.run_id}"
        )

        # Loss TODO generalize
        self.base_loss_fn = nn.CrossEntropyLoss()

        if not self.config.debug:
            wandb_dir = tempfile.mkdtemp()
            LOG.info(f"Writing wandb local logs to {wandb_dir}")

            params = getattr(self.config, 'wandb', {})
            wandb.init(
                config=utils.flatten_dict(self.config),
                name=f"{self.config.dataset.name}/{self.config.model.name}/{self.config.alg.name}_{self.run_id}",
                dir=tempfile.mkdtemp(),
                **params
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

    def echo(self, train_step, info_dict):
        if not self.config.silent:
            LOG.info((
                f"Epoch: {self.epoch}; Train step {train_step}; ",
                f"; ".join([f"{key} {val:0.4f}" for key, val in info_dict.items()])
            ))

    def wandb_log(self, step, info_dict):
        info_dict['step'] = step
        if not self.config.debug:
            wandb.log(info_dict)

    def configure_optimizers(self):
        self.schedulers = []

        self.opt = build_optimizer(self.model.parameters(), self.config.optimizer)
        if self.config.optimizer.use_scheduler:
            scheduler = build_scheduler(self.opt, self.config.optimizer.scheduler)
            self.schedulers.append(scheduler)

    def train_step(self, inputs, labels):
        info_dict = {}
        self.model.train()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(inputs)
        loss = self.base_loss_fn(logits, labels)
        (loss / self.config.accumulate_bs).backward()
        info_dict['loss/train'] = loss.item()

        if self.global_iter % self.config.accumulate_bs == 0:
            self.opt.step()
            self.opt.zero_grad()

        return info_dict

    def val_step(self):
        info_dict = {}
        self.model.eval()

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.config.bs,
            shuffle=True,
            num_workers=2
        )

        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        val_step_count = 0
        with torch.no_grad():
            for val_step, (inputs, labels) in enumerate(val_loader):
                if getattr(self.config, 'max_val_len', None) and val_step >= self.config.max_val_len:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                loss = self.base_loss_fn(logits, labels)

                val_step_count += 1
                total_loss += loss
                # TODO generalize metrics computation
                acc1, acc5 = utils.accuracy(logits, labels, topk=(1, 5))
                total_acc1 += acc1
                total_acc5 += acc5

        info_dict['loss/val'] = (total_loss / val_step_count).item()
        info_dict['acc1'] = (total_acc1 / val_step_count).item()
        info_dict['acc5'] = (total_acc5 / val_step_count).item()

        return info_dict

    def run(self):
        self.configure_optimizers()

        self.model.train()
        self.model.to(self.device)
        self.global_iter = 0

        for epoch in itertools.count():

            self.epoch = epoch
            if getattr(self.config, 'max_epochs', None) and epoch >= self.config.max_epochs:
                return

            train_loader = DataLoader(
                self.train_set,
                batch_size=self.config.bs,
                shuffle=True,
                num_workers=2
            )

            for iter, (inputs, labels) in enumerate(train_loader):
                if getattr(self.config, 'max_iters', None) and self.global_iter >= self.config.max_iters:
                    return

                if getattr(self.config, 'max_iters_per_epoch', None) and iter >= self.config.max_iters_per_epoch:
                    break

                info_dict = self.train_step(inputs, labels)
                if self.global_iter % self.config.log_interval == 0:
                    self.echo(self.global_iter, info_dict)
                    self.wandb_log(self.global_iter, info_dict)

                if self.global_iter % self.config.val_interval == 0:
                    info_dict = self.val_step()
                    self.echo(self.global_iter, info_dict)
                    self.wandb_log(self.global_iter, info_dict)

                self.saveState(self.model, self.global_iter, self.config.model.name)
                self.global_iter += 1

            for scheduler in self.schedulers:
                scheduler.step()

