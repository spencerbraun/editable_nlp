class TrainConfig:
    def __init__(self):
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e4
        self.cedit = 1
        self.cloc = 1

        self.debug = False
        self.model_save_pt = 2500
        self.model_dir = '../models/finetune'

class EditConfig:
    def __init__(self):
        self.inner_lr = 1e-2
        self.outer_lr = 1e-5
        self.epochs = 2
        self.max_training_samps = 2e6
        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 1

        self.debug = True
        self.model_save_pt = 2000
        self.model_dir = '../models'