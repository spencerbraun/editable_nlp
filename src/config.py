class TrainConfig:
    def __init__(self):
        self.outer_lr = 1e-5
        self.epochs = 1
        self.max_training_samps = 1e4
        self.cedit = 1
        self.cloc = 1

        self.debug = False
        self.model_save_pt = 2500
        self.write_loc = '/XXXX'
        self.model_dir = f'{self.write_loc}/models/finetune'

        self.silent = False

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
        self.write_loc = '/XXXX'
        self.model_dir = f'{self.write_loc}/models'

        self.silent = False