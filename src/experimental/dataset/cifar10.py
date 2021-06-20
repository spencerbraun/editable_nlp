import logging
import os
import random
import itertools

import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, RandomSampler
from typing import Any, Callable, Optional, Tuple

from omegaconf import DictConfig
from .base import EditDataset

logger = logging.getLogger(__name__)


class EditCIFAR10(EditDataset, CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

    def edit_generator(self, batch_size):
        """
        Creates an edit generator which returns:
          - outer_data: edit data to evaluate the model on.
          - outer_labels: labels for `outer_data`.
          - inner_data: edit data to train the model on.
          - inner_labels: labels for `inner_data`.
          - loc_data: data to enforce locality.
          - loc_labels: labels for `loc_data`.
        """
        edit_set = CIFAR10(self.root, self.train, self.transform, self.target_transform)
        loc_set = CIFAR10(self.root, self.train, self.transform, self.target_transform)
        edit_loader = DataLoader(
            edit_set,
            sampler=RandomSampler(edit_set),
            batch_size=batch_size
        )
        loc_loader = DataLoader(
            loc_set,
            sampler=RandomSampler(loc_set),
            batch_size=batch_size
        )

        for (edit_images, edit_labels), (loc_images, loc_labels) in zip(itertools.cycle(edit_loader), itertools.cycle(loc_loader)):
            rand_labels = torch.randint_like(edit_labels, len(self.classes))
            yield edit_images, rand_labels, edit_images, rand_labels, loc_images, loc_labels


def cifar10(config: DictConfig):

    CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR10_STD = [0.2023, 0.1994, 0.2010]

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    root = os.path.join(config.base_dir, config.dataset.path)
    params = getattr(config.dataset, 'params', {})

    train_set = EditCIFAR10(root=root, train=True, transform=train_transform, **params)
    logger.info(f"Loaded CIFAR-10 train set from {root}")

    val_set = EditCIFAR10(root=root, train=False, transform=val_transform, **params)
    logger.info(f"Loaded CIFAR-10 validation set from {root}")

    return train_set, val_set

