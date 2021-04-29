class TrainConfig:
    def __init__(self):
        self.task = 'finetune'
        self.model = 'resnet18'
        self.dataset = 'imagenet'
        self.epochs = 1000
        
        self.debug = False
        self.model_save_pt = 100
        self.write_loc = '../..'

        self.silent = False
