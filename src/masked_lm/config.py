class TrainConfig:
    def __init__(self):
        self.task = 'T5_finetune'
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_iter = 1.5e4

        self.debug = True
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

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'
        self.ft_model_name = ""

        self.silent = False
