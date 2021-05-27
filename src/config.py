class TrainConfig:
    def __init__(self):
        self.task = 'finetune'
        self.outer_lr = 1e-5
        self.epochs = 3
        self.max_iter = 1.5e4

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False

class EditConfig:
    def __init__(self):
        self.task = 'editable'
        self.inner_lr = 1e-3
        self.outer_lr = 1e-5
        self.epochs = 3
        self.max_iter = 1.5e4
        self.n_edit_steps = 1
        self.cedit = 0.5
        self.cloc = 10
        self.learnable_lr = True
        self.lr_lr = 1e-2

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False

class SelfSampleGPT2Config:
    def __init__(self):
        self.task = 'gen'
        self.model_name = 'gpt2'
        self.inner_lr = 1e-2
        self.outer_lr = 1e-5
        self.epochs = 3
        self.max_iter = 40000
        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 100
        self.learnable_lr = True
        self.lr_lr = 1e-3

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'
        self.ft_model_name = "gpt2_epoch0_ts10000.20210408.09.04.1617899457"

        self.silent = False
        
        
class ClozeT5Config:
    def __init__(self):
        self.task = 'cloze'
        self.model_name = 't5-small' #t5-small or gpt2
        self.inner_loop = 'template' #sentence, template, or random
        self.max_val_len = 2000

        self.inner_lr = 1e-2
        self.outer_lr = 1e-5
        self.epochs = 3
        self.max_iter = 40000
        self.n_edit_steps = 1
        self.cedit = 5
        self.cloc = 10
        self.learnable_lr = True
        self.lr_lr = 1e-3

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'
        self.ft_model_name = "T5_finetune_epoch0_ts20000.20210505.09.05.1620232583"

        self.silent = False

class LamaBartConfig:
    def __init__(self):
        self.task = 'cloze'
        self.model_name = 'bart-base' #t5-small or gpt2
        self.inner_loop = 'template' #sentence, template, or random
        self.max_val_len = 2000
        
        self.inner_lr = 1e-2
        self.outer_lr = 1e-4
        self.epochs = 2
        self.max_iter = 40000
        self.n_edit_steps = 1
        self.cedit = 5
        self.cloc = 10
        self.learnable_lr = True
        self.lr_lr = 1e-3

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'
        self.ft_model_name = 'bart-base_epoch0_ts285000.2021-05-24_22-15-37-529207byKYd'

        self.silent = False

class KiltBartConfig:
    def __init__(self):
        self.task = 'cloze'
        self.model_name = 'bart-base' #t5-small or gpt2
        self.inner_loop = 'template' #sentence, template, or random
        self.max_val_len = 2000
        
        self.inner_lr = 1e-2
        self.outer_lr = 1e-4
        self.epochs = 2
        self.max_iter = 40000
        self.n_edit_steps = 1
        self.cedit = 5
        self.cloc = 10
        self.learnable_lr = True
        self.lr_lr = 1e-3

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'
        self.ft_model_name = 'bart-base_epoch7_ts40000.2021-05-25_22-13-53-187518OQgeF'

        self.silent = False
