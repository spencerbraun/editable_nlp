import argparse
import os
import random
import copy
from datetime import datetime
import numpy as np

from omegaconf import DictConfig, OmegaConf
from hydra.experimental import compose, initialize_config_dir

import torch
import torch.nn.functional as F
import higher

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                mode=("mmtm" if config.mmtm else "val"),
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

            '''
            total_edit_loss = (
                config.cloc * l_loc  + 
                config.cedit * l_edit
            )
            total_loss = l_base.item() + total_edit_loss.item()
            '''
            total_loss = None

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '-C', '--checkpoint_path', type=str, help='(Relative or absolute) path of the checkpoint')
    parser.add_argument('-s', '-S', '--seq_edits', type=int, default=100)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    checkpoint = os.path.basename(checkpoint_path)  # /path/to/run_dir/models/checkpoint.pth
    checkpoints_dir = os.path.dirname(checkpoint_path)
    run_dir = os.path.dirname(checkpoints_dir)

    config_dir = os.path.join(run_dir, '.hydra')
    initialize_config_dir(config_dir=config_dir, job_name='evaluate')
    config = compose(config_name='config', overrides=[
        f"+run_dir={run_dir}",
        f"+checkpoint_path={checkpoint_path}",
        f"+seq_edits={args.seq_edits}"
    ])

    if hasattr(config, 'seed'):
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if getattr(config, 'deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    dataset_class = getattr(datasets, config.dataset.name)
    train_set, val_set = dataset_class(config)

    if not hasattr(config, 'model'):
        raise ValueError('Config must contain attribute `model`.')
    model = build_model(config)
    editable = EditableModel(model)

    main(config)