import logging
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from typing import Union
from omegaconf import OmegaConf, DictConfig, ListConfig

from experimental.build import build_optimizer, build_scheduler
import torchvision

logger = logging.getLogger(__name__)


def get_logprobs(model, inputs, labels):
    with torch.no_grad():
        outputs = model(inputs)
        lps = F.log_softmax(outputs, dim=-1)

    return lps[range(len(labels)), labels]


def _inner_params(module: nn.Module, edit_modules: Union[str, int, DictConfig, ListConfig], named=False):
    """
    Recursively retrieve (named) parameters from a module as specified by a config `edit_modules`.
    If edit_modules is a
      - str, returns a list of the parameters of `module.edit_modules`
      - int, returns a list of the parameters of the last `edit_modules` layers of `module`.
      - DictConfig, recursively applies `_inner_params(k, v)` to each key-value pair `k, v`
          of the dict and returns the union of the returned parameters.
      - ListConfig, recursively applies `_inner_params(module, elem)` to each element `elem`
          of the list and returns the union of the returned parameters.

    Example: suppose `module` has the following structure:
        module
        |_ transformer
            |_ encoder
                |_ ...
            |_ decoder
                |_ layer1
                |_ layer2
                |_ layer3
                    |_ ...
        |_ head
            |_ layer1
            |_ layer2

    If we wanted to return the parameters of `module.transformer.decoder.layer2`, the final
    5 submodules of `module.transformer.decoder.layer3`, and `module.head`, we could use the
    following yaml config to generate `edit_modules`:
        ```
        # config.yaml
        edit_modules:
        - transformer:
            decoder:
              - layer2
              - layer3: 5
        - head
        ```
    """
    param_fn = 'named_parameters' if named else 'parameters'
    params = []

    if isinstance(edit_modules, str):
        submodule = getattr(module, edit_modules)
        params = list(getattr(submodule, param_fn)())

    elif isinstance(edit_modules, int):
        num_layers = edit_modules
        params = list(getattr(module[-num_layers:], param_fn)())

    elif isinstance(edit_modules, ListConfig):
        for edit_module in edit_modules:
            params += _inner_params(module, edit_module, named)

    elif isinstance(edit_modules, DictConfig):
        for k, v in edit_modules.items():
            submodule = getattr(module, k)
            params += _inner_params(submodule, v, named)

    else:
        raise TypeError(f'`edit_modules` must be of type int, DictConfig, or ListConfig, not {type(edit_modules)}.')

    return params


def _disable_bn_running_stats(module: nn.Module, train: bool = True):
    """
    Disables the tracking and use of running stats in batch-norm submodules.
    """
    def bn_train_fn(self: nn.Module, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = train
        for module in self.children():
            module.train(mode)
        return self
    nn.BatchNorm2d.train = bn_train_fn

    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.train()

    return module


def edit_fn(
    self: nn.Module,
    outer_data: torch.Tensor,
    outer_labels: torch.Tensor,
    inner_data: torch.Tensor,
    inner_labels: torch.Tensor,
    n_edit_steps: int = 1
):
    lp_hist = []
    param_groups = [
        {'params': p, 'lr': None} 
        for p in self.inner_params()
    ]
    edit_opt = build_optimizer(param_groups, self._edit_optimizer_config)

    with torch.enable_grad(), higher.innerloop_ctx(
        self,
        edit_opt,
        override={'lr': self.edit_lrs},
        copy_initial_weights=False,
        track_higher_grads=self.training
    ) as (fmodel, diffopt):

        fmodel.eval()
        for edit_step in range(n_edit_steps):
            if hasattr(fmodel, 'set_editing'):
                fmodel.set_editing(True)

            output = fmodel(inner_data)
            mask = torch.argmax(output, -1) == inner_labels
            # Stop editing early if completely successful
            if mask.all():
                break
            inner_labels_masked = inner_labels.masked_fill(mask, -100)
            loss = fmodel.edit_loss_fn(output, inner_labels_masked).mean() # Include masked elements in the loss mean
            lp_hist.append(get_logprobs(fmodel, outer_data, outer_labels))

            if hasattr(fmodel, 'set_editing'):
                fmodel.set_editing(False)

            diffopt.step(loss, grad_callback=getattr(fmodel, 'grad_callback', None))

        output = fmodel(outer_data)
        mask = torch.argmax(output, -1) == outer_labels
        outer_labels_masked = outer_labels.masked_fill(mask, -100)
        l_edit = fmodel.edit_loss_fn(output, outer_labels_masked).mean()
        lp_hist.append(get_logprobs(fmodel, outer_data, outer_labels))
        edit_success = mask.float().mean() * 100.0

    if self.training:
        model_edited = fmodel
    else:
        model_edited = copy.deepcopy(self)
        model_edited.load_state_dict(fmodel.state_dict())
    model_edited.train(self.training)

    return model_edited, l_edit, lp_hist, edit_success


def editable_wrap_(module: nn.Module, config: DictConfig):
    module._alg_config = config.alg

    if not hasattr(config, 'edit_optimizer'):
        raise ValueError('Config must contain attribute `edit_optimizer`.')

    def inner_params(self: nn.Module):
        params = []
        if not hasattr(self, '_alg_config'):
            logger.warning('Model has no attribute `_alg_config`; defaulting to no inner params.')
        elif self._alg_config.adapt_all:
            params = list(self.parameters())
        else:
            if not hasattr(self._alg_config, 'edit_modules'):
                logger.warning('Algorithm config has no `edit_modules` attribute; continuing with no edit modules.')
            else:
                params = _inner_params(self, module._alg_config.edit_modules, named=False)

        return params

    def named_inner_params(self: nn.Module):
        params = []
        if not hasattr(self, '_alg_config'):
            logger.warning('Model has no attribute `_alg_config`; defaulting to no inner params.')
        elif self._alg_config.adapt_all:
            params = list(self.named_parameters())
        else:
            if not hasattr(self._alg_config, 'edit_modules'):
                logger.warning('Algorithm config has no `edit_modules` attribute; continuing with no edit modules.')
            else:
                params = _inner_params(self, module._alg_config.edit_modules, named=True)

        return params

    if hasattr(type(module), 'inner_params'):
        logger.warning(f'Overriding default `inner_params` implementation of class {type(module)}.')
    type(module).inner_params = inner_params
    if hasattr(type(module), 'named_inner_params'):
        logger.warning(f'Overriding default `named_inner_params` implementation of class {type(module)}.')
    type(module).named_inner_params = named_inner_params

    module.edit_loss_fn = nn.CrossEntropyLoss(reduction='none')
    module._edit_optimizer_config = config.edit_optimizer
    module.edit_lrs = [
        torch.nn.Parameter(torch.tensor(config.edit_optimizer.params.lr)) 
        for p in module.inner_params()
    ]

    if hasattr(type(module), 'edit'):
        logger.warning(f'Overriding default `edit` implementation of class {type(module)}.')
    type(module).edit = edit_fn

    if getattr(config.alg, 'first_order_maml', False):
        def first_order_callback(all_grads):
            detached_grads = []
            for g in all_grads:
                detached_grads.append(g.detach())
            return tuple(detached_grads)
        module.grad_callback = first_order_callback

    # Override default batch-norm behaviors
    _disable_bn_running_stats(module, train=False)


def test_inner_params():
    print("Testing _inner_params implementation")

    edit_modules = OmegaConf.create([
        'conv1',
        'bn1',
        {'layer1': 1},
        {'layer3': {
            '0': [
                'conv2',
                {'downsample' : 2}
            ],
            '1': 'bn1'
        }}
    ])
    model = torchvision.models.resnet18()

    for param_fn in ['parameters', 'named_parameters']:
        true_inner_params = list(itertools.chain(
            getattr(model.conv1, param_fn)(),
            getattr(model.bn1, param_fn)(),
            getattr(model.layer1[-1:], param_fn)(),
            getattr(model.layer3[0].conv2, param_fn)(),
            getattr(model.layer3[0].downsample[-2:], param_fn)(),
            getattr(model.layer3[1].bn1, param_fn)()
        ))
        inner_params = _inner_params(model, edit_modules, named=(param_fn == 'named_parameters'))
        if inner_params != true_inner_params:
            print("FAIL")
        else:
            print("PASS")


if __name__ == '__main__':
    test_inner_params()
