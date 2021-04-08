class TrainConfig:
    def __init__(self):
        self.task = 'finetune'
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e4
        self.cedit = 1
        self.cloc = 1

        self.debug = False
        self.model_save_pt = 2500
        self.write_loc = '..'

        self.silent = False

class EditConfig:
    def __init__(self):
        self.task = 'editable'
        self.inner_lr = 1e-2
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e5
        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 1

        self.debug = True
        self.model_save_pt = 5000
        self.write_loc = '..'

        self.silent = False