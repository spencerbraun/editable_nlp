class TrainConfig:
    def __init__(self):
        self.task = 'finetune'
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1.5e4

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False

class EditConfig:
    def __init__(self):
        self.task = 'editable'
        self.inner_lr = 1e-3
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1.5e4
        self.n_edit_steps = 1
        self.cedit = 0.5
        self.cloc = 5
        self.learnable_lr = True

        self.debug = False
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False

class SelfSampleConfig:
    def __init__(self):
        self.task = 'self_sample'
        self.inner_lr = 1e-3
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1.5e4
        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 10
        self.learnable_lr = True

        self.debug = True
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False
