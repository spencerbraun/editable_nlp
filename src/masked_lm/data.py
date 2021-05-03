import torch
import datasets
from datasets import load_dataset

import utils


class LAMADataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_loc=".", 
        template_filter=None,
        pct=100,
        shuffle=False,
        seed=123,
        mode='finetune'
    ):
        self.dataset = load_dataset(
            'lama', 
            cache_dir=data_loc,
            split=datasets.ReadInstruction('train', to=pct, unit='%'),
            keep_in_memory=True
        )
        self.mode = mode
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=123)
        if template_filter:
            self.dataset = self.dataset.filter(
                 lambda x: x['template'] in template_filter
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        
        sentence = self.dataset['masked_sentence'][index]
        masked_sent = sentence.replace("[MASK]", "<extra_id_0>")
        
        obj_surface = self.dataset['obj_surface'][index]
        orig_label = f"<extra_id_0> {obj_surface.strip()} <extra_id_1>"
        if self.mode == 'finetune':
            return masked_sent, orig_label

        sub_surface = self.dataset['sub_surface'][index]

        template = self.dataset['template'][index]
        template = template.replace("[X]", sub_surface)
        masked_template = template.replace("[Y]", "<extra_id_0>")
        
        edit_surface = obj_surface # TODO: replace with edited entity from same template group
        edit_label = f"<extra_id_0> {edit_surface.strip()} <extra_id_1>"
        
        return masked_sent, masked_template, orig_label, edit_label


class MaskedLMDataloader:
    def __init__(self, dataset, loc, mode='finetune', train_pct=80, **kwargs):
        """
        mode in [finetune, editable]
        kwargs:
            bs: int
            pct: int
            shuffle: bool
            train_pct: int
            max_val_len: int
        """
        
        self.kwargs = kwargs
        self.mode = mode
        
        if dataset.lower() == 'lama':
            self.dataset = LAMADataset(
                data_loc=f"{loc}/hf", 
                template_filter=self.kwargs.get('template_filter'),
                pct=self.kwargs.get('pct', 100),
                shuffle=self.kwargs.get('shuffle', False),
                seed=123,
                mode=self.mode
            )
            
        elif dataset.lower() == 'kilt':
            pass
            
        self.tokenizer = utils.loadT5Tokenizer(cache_dir=loc)
        self.bs = self.kwargs.get('bs', 1)
        
        self.valid_len = int(min(
            (1-train_pct/100) * len(self.dataset), 
            self.kwargs.get('max_val_len', float('inf'))
            ))
        self.train_len = len(self.dataset) - self.valid_len
        self.train_ds, self.valid_ds = torch.utils.data.random_split(
            self.dataset, [self.train_len, self.valid_len]
            )
    
    def pad_collate(self, batch):
        
        out = []        
        for sample in zip(*batch):
            toks = self.tokenizer(
                    sample,
                    truncation=False,
                    padding=True,
                    return_tensors='pt'
                )
            out.append((toks.input_ids, toks.attention_mask))
            
        return out
    
    def getDataloader(self, dataset):
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bs,
            num_workers=2,
            pin_memory=True,
            shuffle=self.kwargs.get('shuffle', False),
            collate_fn=self.pad_collate
        )
        
        return dataloader
    
    @property
    def train(self):
        return self.getDataloader(self.train_ds)

    @property
    def validation(self):
        return self.getDataloader(self.valid_ds)
    