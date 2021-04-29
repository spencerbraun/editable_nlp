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