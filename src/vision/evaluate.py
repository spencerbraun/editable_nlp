import argparse
import os
import glob
import time
import random
import copy
from datetime import datetime
import itertools
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, random_split
import torchvision
from torchvision.models import resnet18, densenet169
import higher

from vision.data_process import loadCIFAR, loadImageNet
import vision.utils as utils
from alg.senn_conditional import ConditionalLinearWrapper


eps = np.finfo(np.float32).eps.item()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_params(model):
    return torch.cat([p.view(-1).detach() for p in model.parameters()])


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
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

    return base_lps[range(labels.shape[0]), labels]


def loadLr(model_path):
    model_name = os.path.basename(model_path)
    model_id = model_name.split(".")[-1]
    step = model_name.split("_")[-1].split(".")[0]
    dir_loc = os.path.dirname(model_path)
    lr_glob = glob.glob(f"{dir_loc}/lr_epoch[0-9]*_{step}.*{model_id}")

    if len(lr_glob) > 1:
        raise AttributeError("Too many lr specifications", ",".join(lr_glob))
    elif len(lr_glob) == 0:
        raise AttributeError("No lr specifications found")
    else:
        print(f"Loading lrs {lr_glob[0]}")
        lrs = torch.load(lr_glob[0])

    return lrs


def repeater(dataloader):
    for loader in itertools.repeat(dataloader):
        for inputs, labels in dataloader:
            yield inputs, labels


def editGenerator(dataset, batch_size=1):
    num_classes = len(dataset.dataset.classes) if type(dataset) == torch.utils.data.Subset else dataset.classes
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        # num_workers=2,
        pin_memory=True
    )
    while True:
        for inputs, labels in repeater(dataloader):
            edit_labels = torch.randint_like(labels, num_classes)
            yield inputs, edit_labels


def performEdits(
    model,
    edit_inputs,
    edit_labels,
    n_edit_steps=1,
    lrs=None,
    default_lr=1e-5,
    mode="train",
    split_params=False
):
    INNER_GRAD_NORMS = []
    MAX_NORM = 10000

    def clip_callback(all_grads):
        nonlocal INNER_GRAD_NORMS
        norm = torch.cat([g.flatten() for g in all_grads]).norm().detach()
        ratio = MAX_NORM / (norm + 1e-6)
        INNER_GRAD_NORMS.append(norm.item())
        return [g * min(1, ratio) for g in all_grads]

    def callback(all_grads):
        nonlocal INNER_GRAD_NORMS
        norm = torch.cat([g.flatten() for g in all_grads]).norm().detach()
        INNER_GRAD_NORMS.append(norm.item())
        return all_grads

    lp_hist = []
    param_groups = [
        {'params': p, 'lr': None} 
        for p in model.inner_params()
    ]
    inner_opt = (
        torch.optim.SGD(param_groups) if lrs
        else torch.optim.SGD(
            model.inner_params(),
            lr=default_lr
        )
    )

    with torch.enable_grad():
        with higher.innerloop_ctx(
            model,
            inner_opt,
            override={'lr': lrs} if lrs else None,
            copy_initial_weights=False,
            track_higher_grads=(mode == "train")
        ) as (fmodel, diffopt):
            fmodel.eval()
            lp_hist.append(get_logprobs(fmodel, edit_inputs, edit_labels).exp())

            for edit_step in range(n_edit_steps):
                if split_params:
                    fmodel.set_editing(True)
                edit_logits = fmodel(edit_inputs)
                loss = F.cross_entropy(edit_logits, edit_labels)
                if split_params:
                    fmodel.set_editing(False)

                diffopt.step(loss, grad_callback=callback)
                lp = get_logprobs(fmodel, edit_inputs, edit_labels)
                lp_hist.append(lp.exp())

    edit_logits = fmodel(edit_inputs)
    l_edit = F.cross_entropy(edit_logits, edit_labels)
    edit_success = (torch.argmax(edit_logits, -1) == edit_labels).float().mean().item() * 100.0
    prob_change = (abs(lp_hist[-1]) - abs(lp_hist[0]))#  / (abs(lp_hist[0]) + eps))
    print("Mean prob history:", ["{:.2f}".format(i.mean().item()) for i in lp_hist])
    print("Edit step {}\tMean prob change {:.2f}\tMean prob {:.2f}\tLoss {:.2f}".format(
            edit_step, prob_change.mean().item(), lp_hist[-1].mean().item(), l_edit))

    if mode == 'train':
        model_edited = fmodel
    else:
        model_edited = copy.deepcopy(model)
        model_edited.load_state_dict(fmodel.state_dict())

    return model_edited, l_edit, lp_hist, prob_change, edit_success, INNER_GRAD_NORMS


def evaluateOnDataset(model, dataset):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2)

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        logits = model(inputs)
        acc1, acc5 = accuracy(logits, labels, topk=(1,5))
        top1.update(acc1, inputs.shape[0])
        top5.update(acc5, inputs.shape[0])

    return top1.avg, top5.avg


def evalEditable(
    model,
    dataset,
    config
):

    savedir = os.path.join(config.run_dir, 'eval')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = f"edit_success_{os.path.basename(config.checkpoint_path)}"
    saveloc = os.path.join(savedir, filename)

    n_edits = 0
    edit_number = 0
    model_number = 0
    sequential = config.seq_edits > 1

    model.to(DEVICE)
    original_model = copy.deepcopy(model)
    orig_params = get_params(original_model)

    try:
        lrs = loadLr(config.checkpoint_path)
    except AttributeError:
        print("Did not load lrs")
        lrs = []

    try:
        n_edit_examples = len(dataset) // 10
        val_dataset, edit_dataset = random_split(dataset, [len(dataset) - n_edit_examples, n_edit_examples])
    except:
        print(f"Not enough validation data to perform {n_edit_examples} edits")

    edit_generator = editGenerator(edit_dataset, config.edit_bs)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=True, num_workers=2)

    with torch.no_grad(), open(saveloc, "w") as f:
        f.write(
            "model_number,edit_number,train_step,n_edit_steps,edit_step,log_prob,"
            "loss,orig_acc1,new_acc1,orig_acc5,new_acc5,norm\n"
            )
        for train_step, (inputs, labels) in enumerate(repeater(val_loader)):
            if train_step >= (5 * config.seq_edits):
                break

            base_inputs, loc_inputs = torch.split(inputs, [(inputs.shape[0] + 1) // 2, inputs.shape[0] // 2])
            base_labels, loc_labels = torch.split(labels, [(labels.shape[0] + 1) // 2, labels.shape[0] // 2])
            edit_inputs, edit_labels = next(edit_generator)

            base_inputs, base_labels = base_inputs.to(DEVICE), base_labels.to(DEVICE)
            loc_inputs, loc_labels = loc_inputs.to(DEVICE), loc_labels.to(DEVICE)
            edit_inputs, edit_labels = edit_inputs.to(DEVICE), edit_labels.to(DEVICE)

            model.eval()
            base_logits = model(base_inputs)
            l_base = F.cross_entropy(base_logits, base_labels)

            model_edited, l_edit, lp_hist, prob_change, edit_success, inner_grad_norms = performEdits(
                model,
                edit_inputs,
                edit_labels,
                n_edit_steps=config.n_edit_steps,
                lrs=lrs,
                default_lr=1e-3,
                mode="val",
                split_params=config.split_params
            )

            model_edited.eval()
            loc_logits = model(loc_inputs)
            edited_loc_logits = model_edited(loc_inputs)
            l_loc = (
                loc_logits.softmax(-1) * 
                (
                    loc_logits.log_softmax(-1) - 
                    edited_loc_logits.log_softmax(-1)
                )
            ).sum(-1).mean()

            total_edit_loss = (
                config.cloc * l_loc  + 
                config.cedit * l_edit
            )
            total_loss = l_base.item() + total_edit_loss.item()

            if edit_number % 20 == 0:
                orig_acc1, orig_acc5 = evaluateOnDataset(model, val_dataset)
                new_acc1, new_acc5 = evaluateOnDataset(model_edited, val_dataset)
            else:
                orig_acc1 = orig_acc5 = new_acc1 = new_acc5 = ""

            norm_diff = orig_params.sub(get_params(model_edited)).norm().item()
            model = model_edited

            for idx, val in enumerate(lp_hist):
                run = (
                    model_number, edit_number, train_step, config.n_edit_steps, idx, val, 
                    total_loss, orig_acc1, new_acc1, orig_acc5, new_acc5, norm_diff
                )
                form = lambda x: str(x.mean().cpu().item()) if torch.is_tensor(x) else str(x)
                writeStr = ",".join([form(x) for x in run])
                f.write(f"{writeStr}\n")

            if edit_number < (config.seq_edits - 1):
                edit_number += 1
            else:
                edit_number = 0
                model_number += 1
                model.load_state_dict(original_model.state_dict())

    print(f"Logged to {saveloc}")


def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))

    loc = utils.sailPreprocess()
    dataset = loadCIFAR(loc, 'val') if config.dataset == 'cifar10' else loadImageNet(loc, 'val')
    num_classes = len(dataset.classes)

    load_model = densenet169 if config.model == 'densenet169' else resnet18
    model = utils.loadOTSModel(load_model, num_classes, config.pretrained)

    if config.model == 'resnet18':
        utils.prep_resnet_for_maml(model)
    elif config.model == 'densenet169':
        utils.prep_densenet_for_maml(model)

    # TODO make compatible with DenseNet
    if config.split_params:
        basic_block_predicate = lambda m: isinstance(m, torchvision.models.resnet.BasicBlock)
        n_hidden = lambda m: m.conv2.weight.shape[0]
        ConditionalLinearWrapper.wrap_model(model, n_hidden, -3, basic_block_predicate)

    if not config.pretrained and not OmegaConf.is_missing(config, 'model_path'):
        model_path = config.checkpoint_path
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    evalEditable(
        model,
        dataset,
        config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', type=str, help='(Relative or absolute) path of the checkpoint')
    parser.add_argument('-S', '--seq_edits', type=int, default=100)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    checkpoint = os.path.basename(checkpoint_path)  # /path/to/run_dir/checkpoints/checkpoint.pth
    checkpoints_dir = os.path.dirname(checkpoint_path)
    run_dir = os.path.dirname(checkpoints_dir)

    config_dir = os.path.join(run_dir, '.hydra')
    initialize_config_dir(config_dir=config_dir, job_name='evaluate')
    config = compose(config_name='config', overrides=[
        f"+run_dir={run_dir}",
        f"+checkpoint_path={checkpoint_path}",
        f"+seq_edits={args.seq_edits}"
    ])

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    main(config)

