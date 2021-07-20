import logging
import os
import random
import itertools

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, RandomSampler
from typing import Any, Callable, Optional

from omegaconf import DictConfig
from .base import EditDataset

logger = logging.getLogger(__name__)


class EditImageNet(EditDataset, ImageFolder):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.is_valid_file = is_valid_file

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
        edit_set = ImageFolder(self.root, self.transform, self.target_transform, self.loader, self.is_valid_file)
        loc_set = ImageFolder(self.root, self.transform, self.target_transform, self.loader, self.is_valid_file)
        edit_loader = DataLoader(
            edit_set,
            sampler=RandomSampler(edit_set),
            batch_size=batch_size,
        )
        loc_loader = DataLoader(
            loc_set,
            sampler=RandomSampler(loc_set),
            batch_size=batch_size,
        )

        for (edit_images, edit_labels), (loc_images, loc_labels) in zip(edit_loader, loc_loader):
            rand_labels = torch.randint_like(edit_labels, len(self.classes))
            yield edit_images, rand_labels, edit_images, rand_labels, loc_images, loc_labels


def imagenet(config: DictConfig):

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    root = os.path.join(config.base_dir, config.dataset.path)
    params = getattr(config.dataset, 'params', {})

    train_set = EditImageNet(root=os.path.join(root, 'train'), transform=train_transform, **params)
    print(f"Loaded ImageNet train set from {root}")

    val_set = EditImageNet(root=os.path.join(root, 'val'), transform=val_transform, **params)
    print(f"Loaded ImageNet validation set {root}")

    return train_set, val_set

