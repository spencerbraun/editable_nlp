import random

import torch
import torch.nn.functional as F
import higher

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def get_logprobs(model, inputs, labels):
    with torch.no_grad():
        base_logits = model(inputs)
        base_lps = F.log_softmax(base_logits, dim=-1).detach().cpu()
    
    return base_lps[:, labels]





