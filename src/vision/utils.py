
import os
import torch
import torchvision.models as models
import numpy as np

eps = np.finfo(np.float32).eps.item()

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
        self.avg = self.sum / (self.count + eps)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def loadOTSModel(loadModel=models.resnet18, num_classes=1000, pretrained=True):
    model = loadModel(num_classes=num_classes, pretrained=pretrained)
    if pretrained:
        print("Loaded pretrained model")
    return model


def loadTrainedModel(modelPath, loadModel=models.resnet18, num_classes=1000):
    model = loadModel(num_classes=num_classes)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    print(f"Loaded checkpoint from {modelPath}")
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
