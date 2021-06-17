import logging
import os
import torch
import torchvision

import hydra
from omegaconf import DictConfig
from alg.senn_conditional import ConditionalLinearWrapper

logger = logging.getLogger(__name__)


def build_model(config: DictConfig):
    if not hasattr(config.model, 'backend'):
        raise ValueError('Model config must contain attribute `backend`.')

    if config.model.backend == 'torchvision':
        if not hasattr(torchvision.models, config.model.name):
            raise ValueError('Model name must be defined in `torchvision.models`.')
        model_class = getattr(torchvision.models, config.model.name)

        params = getattr(config.model, 'params', {})
        model = model_class(**params)

        # TODO Wrap model here
        # if getattr(config.alg, 'split_params', False) and getattr(config.model, 'wrapped', False):
        #     wrapping logic

        if getattr(config.model, 'path', None):
            model_path = os.path.join(config.base_dir, config.model.path)
            model.load_state_dict(torch.load(model_path))
            logger.info(f'Loaded state dict from {model_path}')

        # if getattr(config.alg, 'split_params', False) and not getattr(config.model, 'wrapped', False):
        #     wrapping logic

    elif config.model.backend == 'huggingface':
        # TODO
        raise NotImplementedError

    if config.alg.name in ['enn', 'senn']:
        from experimental.models import editable_wrap_
        editable_wrap_(model, config)

    return model


def build_optimizer(parameters, optimizer_config: DictConfig):
    if not hasattr(optimizer_config, 'type'):
        raise ValueError('Optimizer config must contain attribute `type`.')
    optimizer_type = optimizer_config.type

    if not hasattr(torch.optim, optimizer_type):
        raise ValueError('Optimizer type must be defined in `torch.optim`.')
    optimizer_class = getattr(torch.optim, optimizer_type)

    if not hasattr(optimizer_config, 'params'):
        logger.warning('Optimizer config has no parameters')
    params = getattr(optimizer_config, 'params', {})

    opt = optimizer_class(parameters, **params)
    return opt


def build_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: DictConfig):
    if not hasattr(scheduler_config, 'type'):
        raise ValueError('Scheduler config must contain attribute `type`.')
    scheduler_type = scheduler_config.type

    if not hasattr(torch.optim.lr_scheduler, scheduler_type):
        raise ValueError('Scheduler type must be defined in `torch.optim.lr_scheduler`.')
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)

    if not hasattr(scheduler_config, 'params'):
        logger.warning('Scheduler config has no parameters')
    params = getattr(scheduler_config, 'params', {})

    scheduler = scheduler_class(optimizer, **params)
    return scheduler
