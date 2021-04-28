import argparse

import torch
from torchvision import datasets, transforms as T
from PIL import Image


def getCIFARTrain():
    transform = T.Compose([
        T.RandomCrop((24, 24)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar10_data = datasets.CIFAR10('./data',train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(cifar10_data, batch_size=128, shuffle=True, num_workers=2)
    return dataloader


def getCIFARVal():
    transform = T.Compose([
        T.CenterCrop((24, 24)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar10_data = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(cifar10_data, batch_size=128, shuffle=True, num_workers=2)
    return dataloader


def getImageNetTrain():
    transform = T.Compose([
        T.RandomSizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = datasets.ImageNet('./data', split='train', transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True, num_workers=2)
    return dataloader


def getImageNetVal():
    transform = T.Compose([
        T.Scale(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagenet_data = datasets.ImageNet('./data', split='val', transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=4, shuffle=True, num_workers=2)
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar', action='store_true')
    args = parser.parse_args()

    if args.cifar:
        dataloader = getCIFARTrain()
    else:
        dataloader = getImageNetTrain()


