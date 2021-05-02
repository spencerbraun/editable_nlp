class TrainConfig:
    def __init__(self):
        self.task = 'finetune'
        self.model = 'resnet18'
        self.dataset = 'imagenet'
        self.epochs = 200 
        self.outer_lr = 1e-3
        
        self.debug = False
        self.model_save_pt = 1000
        self.write_loc = '../..'

        self.silent = False

class EditConfig:
    def __init__(self):
        self.task = 'editable'
        self.model = 'resnet18'
        self.dataset = 'imagenet'
        self.inner_lr = 1e-2
        self.outer_lr = 1e-3
        self.epochs = 100

        self.n_edit_steps = 1
        self.cedit = 1
        self.cloc = 0.01
        self.learnable_lr = True
        self.lr_lr = 1e-2
        
        self.debug = False
        self.model_save_pt = 1000
        self.write_loc = '../..'

        self.silent = False
