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
