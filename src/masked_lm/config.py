class TrainConfig:
    def __init__(self):
        self.task = 't5_finetune'
        self.model_name = 't5-small'
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_iter = 20000

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False


class SelfSampleConfig:
    def __init__(self):
        self.task = 'T5_self_sample'
        self.model_name = 't5-small' #t5-small or gpt2
        self.inner_loop = 'template' #sentence, template, or random

        self.inner_lr = 1e-2
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_iter = 20000
        self.n_edit_steps = 1
        self.cedit = 0.5
        self.cloc = 1
        self.learnable_lr = True
        self.lr_lr = 1e-3

        self.debug = True
        self.model_save_pt = 100
        self.write_loc = '..'
        self.ft_model_name = "T5_finetune_epoch0_ts100.20210502.19.05.1620009939"

        self.silent = False
