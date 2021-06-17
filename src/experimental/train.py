import torch
import numpy as np
import random

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict

from experimental.models import editable_wrap_
from experimental.build import build_model
from experimental.trainers import BaseTrainer, EditTrainer
import experimental.utils as utils
import experimental.datasets as datasets


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):

    base_dir = utils.sailPreprocess()
    OmegaConf.set_struct(config, True)
    with open_dict(config):
        config.base_dir = base_dir

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'seed'):
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if getattr(config, 'deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    if not hasattr(config, 'dataset'):
        raise ValueError('Config must contain attribute `dataset`.')

    if not hasattr(config.dataset, 'name'):
        raise ValueError('Dataset config must contain attribute `name`.')
    if not hasattr(datasets, config.dataset.name):
        raise ValueError('Dataset must be defined in `experimental.datasets`.')

    dataset_class = getattr(datasets, config.dataset.name)
    train_set, val_set = dataset_class(config)

    if not hasattr(config, 'model'):
        raise ValueError('Config must contain attribute `model`.')
    model = build_model(config)

    if config.alg.name in ['enn', 'senn']:
        trainer = EditTrainer(model, config, train_set, val_set)

    else:
        trainer = BaseTrainer(model, config, train_set, val_set)

    trainer.run()


if __name__ == '__main__':
    main()

