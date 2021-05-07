import os
import argparse
import random
import pickle
import time
from tqdm import tqdm

import torch
import datasets
from datasets import load_dataset

import sys
sys.path.append("..")
import utils


class LAMADataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_loc=".", 
        template_filter=None,
        pct=100,
        shuffle=False,
        seed=123,
        mode='finetune',
        inner_loop='template'
    ):
        self.data_loc = data_loc
        self.pct = pct
        self.inner_loop = inner_loop
        self.mode = mode
        
        self.edit_location = f"{data_loc}/lama_edited_{self.pct}pct.pkl"
        if os.path.exists(self.edit_location):
            print(f"Reusing edited LAMA: {self.edit_location}")
            with open(self.edit_location, "rb") as f:
                self.dataset = pickle.load(f)
        else: 
            print(f"Edited LAMA not found ('{self.edit_location}'). Generating in 5s...")
            time.sleep(5)
            print("Generating...")
            self.generateEdited(self.data_loc)
                

    def tokenize(self, text):
        tok = self.tokenizer(
            text,       
            truncation=False,
            padding=True,
            return_tensors='pt'
        )
        return tok["input_ids"], tok["attention_mask"]
        
    def generateEdited(self, loc):
        random.seed(123)

        dataset = load_dataset(
                'lama', 
                'trex',
                cache_dir=self.data_loc,
                split=datasets.ReadInstruction('train', to=self.pct, unit='%')
            )

        dataset = dataset.shuffle(123)
        templates = list(set(dataset['template']))
        edit_dict = {k: [] for k in templates}
        for temp, obj in zip(dataset['template'], dataset['obj_surface']):
            edit_dict[temp].append(obj)

        edit_dict = {k: list(set(v)) for k, v in edit_dict.items()}

        features = ['masked_sentence', 'template', 'obj_surface', 'sub_surface']
        data_out = {'edit_surface':[]}
        for feat in features:
            data_out[feat] = dataset[feat]

        for idx in tqdm(range(len(dataset))):
            temp = data_out['template'][idx]
            obj = data_out['obj_surface'][idx]
            while True:
                edit = random.choice(edit_dict[temp])
                if edit != obj:
                    break

            data_out['edit_surface'].append(edit)

        with open(self.edit_location, "wb") as f:
            pickle.dump(data_out, f)
            print(f'Edited LAMA written to "{self.edit_location}"')
            self.dataset = data_out
            print("self.dataset populated")
    
    def processMasks(self, idx, kind):
        if kind == 'sentence':
            sentence = self.dataset['masked_sentence'][idx]
            to_return = sentence.replace("[MASK]", "<extra_id_0>")
        elif kind == 'label':
            obj_surface = self.dataset['obj_surface'][idx]
            to_return = f"<extra_id_0> {obj_surface.strip()} <extra_id_1>"

        return self.tokenize(to_return)

    def __iter__(self):
        return (self[idx] for idx in range(len(self)))

    def __len__(self):
        return len(self.dataset['masked_sentence'])

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        edit_idxs = rng.choice(self.list_IDs, self.n_edits, replace=False)
        loc_idxs = rng.choice(len(self.wiki), self.n_edits, replace=False)
        base_idxs = rng.choice(len(self.wiki), self.batch_size, replace=False)
        
        original_sent, edited_sent = [], []
        original_label, edited_label= [], []
        for idx in edit_idxs:
            masked_sent = self.processMasks(idx, 'sentence') 
            orig_label = self.processMasks(idx, 'label')
            
            sub_surface = self.dataset['sub_surface'][idx]
            template = self.dataset['template'][index]
            template = template.replace("[X]", sub_surface)
            masked_template = template.replace("[Y]", "<extra_id_0>")
            
            edit_surface = self.dataset['edit_surface'][idx]
            edit_label = f"<extra_id_0> {edit_surface.strip()} <extra_id_1>"

            original_sent.append(masked_sent)
            original_label.append(orig_label)
            edited_sent.append(self.tokenize(masked_template))
            edited_label.append(self.tokenize(edit_label))

        original_tokens, original_mask = tuple(zip(*original_sent))
        edited_tokens, edited_mask = tuple(zip(*edited_sent))

        original_labels, original_lab_mask = tuple(zip(*original_label))
        edited_labels, edited_lab_mask = tuple(zip(*edited_label))

        loc_tokens, loc_mask = tuple(zip(*[
           self.processMasks(idx, 'sentence') 
            for idx in loc_idxs
            ]))
        loc_labels, loc_lab_mask = tuple(zip(*[
            self.processMasks(idx, 'label') 
            for idx in loc_idxs
            ]))

        base_tokens, base_mask = tuple(zip(*[
            self.processMasks(idx, 'sentence') 
            for idx in base_idxs
            ]))
        base_labels, base_lab_mask = tuple(zip(*[
            self.processMasks(idx, 'label') 
            for idx in base_idxs
            ]))

        to_return = [
            base_tokens, base_mask, base_labels, 
            loc_tokens, loc_mask, loc_labels, 
            original_tokens, original_mask, edited_labels, #sentence and template labels should be the same
            edited_tokens, edited_mask, edited_labels
            ]
        to_return = [torch.cat(tensors) for tensors in to_return]
        
        if self.mode == 'finetune':
            return masked_sent, orig_label

        return to_return


class MaskedLMDataloader:
    def __init__(self, dataset, loc, mode, train_pct=80, **kwargs):
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
        return self.train_ds

    @property
    def validation(self):
        return self.valid_ds
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', default=False)
    args = parser.parse_args()
    
    loc = utils.sailPreprocess()
    
    if args.generate:
        generateEdited(loc)