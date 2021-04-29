
import os
import torch
import torchvision.models as models


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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def loadOTSModel(loadModel=models.resnet18, num_classes=1000, pretrained=True):
    model = loadModel(num_classes=num_classes, pretrained=pretrained)
    return model


def loadTrainedModel(modelPath, loadModel=models.resnet18, num_classes=1000):
    model = loadModel(num_classes=num_classes)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
            res.append(correct_k * 100.0 / batch_size)
        return res


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
