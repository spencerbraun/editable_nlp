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
from .base import BaseTrainer

logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


class EditTrainer(BaseTrainer):

    def __init__(self, model, config: DictConfig, train_set: Dataset, val_set: Dataset):

        super().__init__(model, config, train_set, val_set)

        self.edit_gen = self.train_set.edit_generator(batch_size=config.edit_bs)
        self.loc_loss_fn = lambda input, target: (
            input.softmax(-1) * (
                input.log_softmax(-1) - target.log_softmax(-1)
            )
        ).sum(-1).mean()

    def configure_optimizers(self):
        # Configure base optimizer and schedulers
        super().configure_optimizers()

        if self.config.alg.learnable_lr:
            if not hasattr(self.config, 'lr_optimizer'):
                raise ValueError('`learnable_lr` is set to True but no learning rate optimizer config was found.')

            self.lr_opt = build_optimizer(self.model.edit_lrs, self.config.lr_optimizer)
            if self.config.optimizer.use_scheduler:
                lr_scheduler = build_scheduler(self.lr_opt, self.config.optimizer.scheduler)
                self.schedulers.append(lr_scheduler)

    def train_step(self, inputs, labels):
        info_dict = {}
        self.model.train()

        total_l_edit = 0.0
        total_l_loc = 0.0
        for edit in range(self.config.n_edits):

            outer_data, outer_labels, inner_data, inner_labels, loc_data, loc_labels = next(self.edit_gen)
            outer_data = outer_data.to(self.device)
            outer_labels = outer_labels.to(self.device)
            inner_data = inner_data.to(self.device)
            inner_labels = inner_labels.to(self.device)

            model_edited, l_edit, lp_hist, edit_success = self.model.edit(
                outer_data,
                outer_labels,
                inner_data,
                inner_labels,
                self.config.n_edit_steps
            )
            total_l_edit += l_edit / self.config.n_edits

            loc_data = loc_data.to(self.device)
            loc_labels = loc_labels.to(self.device)

            loc_logits = self.model(loc_data).detach()
            edited_loc_logits = model_edited(loc_data)
            l_loc = self.loc_loss_fn(loc_logits, edited_loc_logits)
            total_l_loc += l_loc / self.config.n_edits

        total_edit_loss = self.config.cedit * total_l_edit + self.config.cloc * total_l_loc
        (total_edit_loss / self.config.accumulate_bs).backward()

        info_dict['loss/edit'] = total_l_edit.item()
        info_dict['loss/loc'] = total_l_loc.item()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        base_logits = self.model(inputs)
        l_base = self.base_loss_fn(base_logits, labels)
        (l_base / self.config.accumulate_bs).backward()
        info_dict['loss/base'] = l_base.item()

        if self.global_iter % self.config.accumulate_bs == 0:
            self.opt.step()
            self.opt.zero_grad()

            if self.config.alg.learnable_lr:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.data

        return info_dict

    def val_step(self):
        info_dict = {}
        self.model.eval()

        for factor in (1, 2, 4):
            val_loader = DataLoader(
                self.val_set,
                batch_size=self.config.bs,
                shuffle=True,
                num_workers=2,
            )
            val_edit_gen = self.val_set.edit_generator(batch_size=factor * self.config.edit_bs)

            total_loss = 0.0
            total_acc1_pre = 0.0
            total_acc5_pre = 0.0
            total_acc1_post = 0.0
            total_acc5_post = 0.0
            total_edit_success = 0.0
            val_step_count = 0
            with torch.no_grad():
                for val_step, (inputs, labels) in enumerate(val_loader):
                    if getattr(self.config, 'max_val_len', None) and val_step >= self.config.max_val_len:
                        break

                    outer_data, outer_labels, inner_data, inner_labels, loc_data, loc_labels = next(val_edit_gen)
                    outer_data = outer_data.to(self.device)
                    outer_labels = outer_labels.to(self.device)
                    inner_data = inner_data.to(self.device)
                    inner_labels = inner_labels.to(self.device)

                    model_edited, l_edit, lp_hist, edit_success = self.model.edit(
                        outer_data,
                        outer_labels,
                        inner_data,
                        inner_labels,
                        self.config.n_edit_steps
                    )

                    loc_data = loc_data.to(self.device)
                    loc_labels = loc_labels.to(self.device)

                    loc_logits = self.model(loc_data)
                    edited_loc_logits = model_edited(loc_data)
                    l_loc = self.loc_loss_fn(loc_logits, edited_loc_logits)

                    total_edit_loss = (self.config.cedit * l_edit + self.config.cloc * l_loc)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    base_logits = self.model(inputs)
                    l_base = self.base_loss_fn(base_logits, labels)
                    edited_base_logits = model_edited(inputs)

                    loss = l_base + total_edit_loss

                    val_step_count += 1
                    total_loss += loss
                    # TODO generalize metrics computation
                    acc1_pre, acc5_pre = utils.accuracy(base_logits, labels, topk=(1, 5))
                    acc1_post, acc5_post = utils.accuracy(edited_base_logits, labels, topk=(1, 5))
                    total_acc1_pre += acc1_pre
                    total_acc5_pre += acc5_pre
                    total_acc1_post += acc1_post
                    total_acc5_post += acc5_post
                    total_edit_success += edit_success

            info_dict[f'loss/val_{factor}'] = (total_loss / val_step_count).item()
            info_dict[f'acc1_pre_{factor}'] = (total_acc1_pre / val_step_count).item()
            info_dict[f'acc5_pre_{factor}'] = (total_acc5_pre / val_step_count).item()
            info_dict[f'acc1_post_{factor}'] = (total_acc1_post / val_step_count).item()
            info_dict[f'acc5_post_{factor}'] = (total_acc5_post / val_step_count).item()
            info_dict[f'edit_success_{factor}'] = (total_edit_success / val_step_count).item()

        return info_dict

