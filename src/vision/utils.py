
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-10)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def prep_resnet_for_maml(model, adapt_all: bool = False, layers=[3]):
    """
    Adapt ResNet model for MAML training.
    """
    # Default inner loop adaptation parameters
    def _inner_params(self):
        if adapt_all:
            return list(self.parameters())
        else:
            params = []
            for l in layers:
                params.extend(getattr(self, f"layer{l}").parameters())
            assert len(params) > 0
            return params

    for mod in model.modules():
        if type(mod) == nn.BatchNorm2d:
            mod.track_running_stats = False

    type(model).inner_params = _inner_params


def prep_densenet_for_maml(model, adapt_all: bool = False):
    # Default inner loop adaptation parameters
    def _inner_params(self):
        if adapt_all:
            return list(self.parameters())
        else:
            return list(self.features.denseblock3.parameters())

    type(model).inner_params = _inner_params


def loadOTSModel(model='resnet18', num_classes=1000, pretrained=True, layernorm=False):
    def _recursive_apply(module: nn.Module):
        n_replaced = 0
        for idx, (name, mod) in enumerate(module.named_children()):
            if type(mod) == nn.BatchNorm2d:
                num_features = mod.num_features
                setattr(module, name, nn.GroupNorm(1, num_features))
                n_replaced += 1
            else:
                n_replaced += _recursive_apply(mod)

        return n_replaced

    load_model = getattr(torchvision.models, model)
    model = load_model(num_classes=num_classes, pretrained=pretrained)
    if pretrained:
        print("Loaded pretrained model")

    if layernorm:
        n_replaced = _recursive_apply(model)
        print(f"Replaced {n_replaced} BatchNorm layers with LayerNorm")

    return model


def sailPreprocess():
    user = os.environ["USER"]
    if user == "spencerb":
        machine_name = platform.node().split(".")[0]
        scr = max(os.listdir(f"/{machine_name}"))
        save_loc = f"/{machine_name}/{scr}"
        local_dir = f"{save_loc}/{user}"

    else:
        save_loc = "/iris/u"
        local_dir = f"{save_loc}/{user}/code/editable_nlp"

    return local_dir
