import os
import argparse

import torch
from torchvision import datasets, transforms as T

import utils


def loadCIFAR(root='../..', split='train', batch_size=128):
    """
    Loads and preprocesses the CIFAR-10 dataset.
    Params:
        - root (str): root directory where the data directory is located.
        - split (str): takes values 'train' and 'val'.
    """
    data_path = os.path.join(root, 'data')
    if split == 'train':
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = True
    elif split == 'val':
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = False
    else:
        msg = f"Split must be train or val, not {split}."
        raise RuntimeError(msg)

    cifar10_data = datasets.CIFAR10(data_path, train=train, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(cifar10_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def loadImageNet(root='../..', split='train', batch_size=4):
    """
    Loads and preprocesses the ImageNet dataset.
    Params:
        - root (str): root directory where the data directory is located.
        - split (str): takes values 'train' and 'val'.
    """
    data_path = os.path.join(root, 'data/ImageNet')
    if split == 'train':
        transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = True
    elif split == 'val':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = False
    else:
        msg = f"Split must be train or val, not {split}."
        raise RuntimeError(msg)

    data_path = os.path.join(data_path, split)
    imagenet_data = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def loadImageNetVal():
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = datasets.ImageFolder('./data/ImageNet/val', transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True, num_workers=2)
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    args = parser.parse_args()

    root = utils.sailPreprocess()
    split = 'train' if args.train else 'val'
    if args.cifar:
        dataloader = loadCIFAR(root, split)
    else:
        dataloader = loadImageNet(root, split)


