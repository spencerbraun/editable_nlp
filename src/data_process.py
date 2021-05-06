import argparse
import os
import collections
import random
import time
from operator import itemgetter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
from datasets import load_dataset, list_metrics, load_metric
from tqdm import tqdm

import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def filterText(iterator):

    valid  = []
    for text in iterator:
        if len(text.split(' ')) < 50:
            continue
        if not is_ascii(text):
            continue
        valid.append(text)

    return valid



class DataProcessor:
    def __init__(
        self, 
        text, 
        write_dir=None
    ):
        self.text = text

        self.write_dir = write_dir
        
        self.raw_texts = []
        self.ner_texts = []
        self.permuted = []
        self.changed_ents = []

        self.output_dict = {}
        
        self.ents = collections.defaultdict(list)

        self.model = spacy.load(
            "en_core_web_md", 
            exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        )
        self.model.add_pipe('sentencizer')
        
        self.keep_ents = ['PERSON']
        
   
    def run(self, func, args):
        for output in func(*args):
            yield output
    
    def permuteEnts(self):
        timestamp = time.time()
            
        for idx, (sent, ents) in enumerate(self.ner_texts):
            
            if self.write_dir:
                if not os.path.exists(self.write_dir):
                    os.makedirs(self.write_dir)
                    print(f"Warning: {self.write_dir} does not exist. Creating...")
                # permuteFile = open(self.write_dir + f'/permuted_entities.{idx}', 'w')
                # origFile = open(self.write_dir + f'/original_entities.{idx}', 'w')
                # entFile = open(self.write_dir + f'/entity_swaps.{idx}', 'w')

            eligible = list(filter(lambda x: x[3] in self.keep_ents, ents))
            orig_ent = random.choice(eligible)
            ent_type = orig_ent[3]
            start, end  = orig_ent[1:3]
            while True:
                replace_ent = random.choice(self.ents[ent_type])
                if replace_ent != orig_ent[0]: break

            prefix = sent[:start]
            suffix = sent[end:]
            new_sent = prefix + replace_ent + suffix

            # if self.write_dir:
                # permuteFile.write(new_sent + "\n")
                # origFile.write(self.raw_texts[idx].strip('\n').strip(" ") + "\n")
                # entFile.write(f"{orig_ent[0]}|{replace_ent}\n")

                # permuteFile.close()
                # origFile.close()
                # entFile.close()

            self.output_dict[idx] = {
                'original': self.raw_texts[idx].strip('\n').strip(" "),
                'edited': new_sent,
                'entity': f"{orig_ent[0]}|{replace_ent}"
            }
                
            self.permuted.append(new_sent)
            self.changed_ents.append((orig_ent[0], replace_ent))
        if self.write_dir:
            torch.save(self.output_dict, self.write_dir + '/data_torch')
            
    
    def processEnts(self):
                
        for output in tqdm(self.runNER(self.text)):
            self.ner_texts.append(output)
        
        
    def runNER(self, texts):
        for doc in tqdm(self.model.pipe(texts)):
            processed = []
            for sent in doc.sents:
                if any([e.label_ in self.keep_ents for e in sent.ents]):
                    ents = []
                    for e in sent.ents:
                        ents.append((e.text, e.start_char - sent.start_char, e.end_char - sent.start_char, e.label_))
                        self.ents[e.label_].append(e.text)
                    processed.append((sent.text, ents))
            if processed:
                self.raw_texts.append(doc.text)
                yield random.choice(processed)
            
    
    def __repr__(self):
        
        return (f"DataProcessor:<{len(self.text)} RAW>"
                f"<{len(self.ner_texts)} NER>"
                f"<{len(self.permuted)} PERM>"
                f"<{sum([len(self.ents[k]) for k in self.ents])} ENTS>")


class NTokenDataset(torch.utils.data.IterableDataset):
    def __init__(self, list_IDs, tokenizer, data_loc="..", dataset='train', max_length=200, batch_size=1, self_sample=False,
                 n_edits=1):
        self.list_IDs = list_IDs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ent_length = 5
        self.dataset = dataset
        self.loc = data_loc
        self.self_sample = self_sample
        self.n_edits = n_edits
        self.wiki = WikitextDataset("/scr/em7", dataset=dataset, filter_=False)
        self.batch_size = batch_size

        if self.dataset == "validation":
            self.dataset = "valid"

    def tokenize(self, text):
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        return tok["input_ids"], tok["attention_mask"]

    def __iter__(self):
        return (self[idx] for idx in range(len(self)))
    
    def __len__(self):
        return len(self.wiki)
    
    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        edit_idxs = rng.choice(self.list_IDs, self.n_edits, replace=False)
        loc_idxs = rng.choice(len(self.wiki), self.n_edits, replace=False)
        base_idxs = rng.choice(len(self.wiki), self.batch_size, replace=False)
        
        path = f"{self.loc}/data/self_sample/{self.dataset}"
        original, edited = [], []
        for idx in edit_idxs:
            with open(f"{path}/original_text.{idx}") as orig:
                original_text = orig.read()
            with open(f"{path}/generated_text.{idx}") as gen:
                generated_text = gen.read()

            original.append(self.tokenize(" " + original_text))
            edited.append(self.tokenize(" " + generated_text))

        original_tokens, original_mask = tuple(zip(*original))
        edited_tokens, edited_mask = tuple(zip(*edited))
        
        loc_tokens, loc_mask = tuple(zip(*[self.tokenize(self.wiki[idx]) for idx in loc_idxs]))
        base_tokens, base_mask = tuple(zip(*[self.tokenize(self.wiki[idx]) for idx in base_idxs]))

        to_return = [base_tokens, base_mask, loc_tokens, loc_mask, original_tokens, original_mask, edited_tokens, edited_mask]
        to_return = [torch.cat(tensors) for tensors in to_return]
        edited_labels = to_return[-2].clone()
        edited_labels[:,:-(5+1)] = -100
        edited_labels[:,-1] = -100
        to_return.append(edited_labels)

        return to_return
    

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, tokenizer, data_loc="..", dataset='train', max_length=200, self_sample=False,
                 n_edits=1):
        self.list_IDs = list_IDs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ent_length = 5
        self.dataset = dataset
        self.loc = data_loc
        self.self_sample = self_sample
        self.n_edits = n_edits

        rng = np.random.default_rng(0)
        self.edit_batches = [rng.choice(self.list_IDs, n_edits, replace=False) for _ in self.list_IDs]

    def tokenize(self, textList, ent=False):
        tokList = []
        for idx in range(len(textList)):
            if ent:
                tok = self.tokenizer(
                    textList[idx],
                    truncation=True,
                    max_length=self.ent_length, 
                    padding="max_length"
                )
            else:
                tok = self.tokenizer(
                    textList[idx],
                    truncation=True,
                    max_length=self.max_length, 
                    padding="max_length"
                )
            tokList.append(
                (
                    torch.tensor(tok['input_ids']), 
                    torch.tensor(tok['attention_mask'])
                )
            )
        if len(tokList) > 1:
            return tokList
        return tokList[0]


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ## TODO: make this specific to a batch 
        
        ID = self.list_IDs[index]
        
        path = f"{self.loc}/data"
        if self.self_sample:
            path = os.path.join(path, 'self_sample')

        if self.dataset == 'train':
            path = os.path.join(path, 'train')
        elif self.dataset == 'validation':
            path = os.path.join(path, 'valid')
        elif self.dataset == 'test':
            path = os.path.join(path, 'test')

        raw, perm, new_ent_tok, old_ent_tok = [], [], [], []
        for idx in self.edit_batches[ID]:
            if self.self_sample:
                with open(f"{path}/original_text.{idx}") as orig:
                    original_text = orig.read()
                with open(f"{path}/generated_text.{idx}") as gen:
                    generated_text = gen.read()

                tokens = self.tokenize([" "+original_text, " "+generated_text])
                raw.append(tokens[0])
                perm.append(tokens[1])
                new_ent_tok.append(-1) # unused
                old_ent_tok.append(-1) # unused
            else:
                with open(f"{path}/original_entities.{idx}") as raw:
                    raw_sample = raw.read()
                with open(f"{path}/permuted_entities.{idx}") as perm:
                    permuted_sample = perm.read()
                with open(f"{path}/entity_swaps.{idx}") as ent:
                    ent_sample = ent.read()
                    ents = ent_sample.strip().split('|')
                    new_ent = ents[-1]
                    old_ent = ents[0]

                tokens = self.tokenize([" "+raw_sample, " "+permuted_sample])
                ent_tokens = self.tokenize([" "+new_ent, " "+old_ent], ent=True)
                raw.append(tokens[0])
                perm.append(tokens[1])
                new_ent_tok.append(ent_tokens[0])
                old_ent_tok.append(ent_tokens[1])

        return raw, perm, new_ent_tok, old_ent_tok


class WikitextDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_loc="/scr", 
        dataset='train', 
        pct=100, 
        min_length=100,
        filter_=True
    ):
        self.dataset = load_dataset(
            'wikitext', 
            'wikitext-103-raw-v1', 
            cache_dir=data_loc, 
            split=f'{dataset}[:{pct}%]'
        )
        print(f"Filtering {len(self.dataset['text'])} wiki items")
        if filter_:
            self.filtered = self.filterText(self.dataset['text'])
        else:
            self.filtered = self.dataset['text']
        print(f"Got {len(self.filtered)} wiki items.")
        self.min_length = min_length
    
    @staticmethod
    def filterText(iterator, min_len=100):
        isascii = lambda s: all(ord(c) < 128 for c in s)
        valid  = []
        for text in iterator:
            if len(text) < min_len:
                continue
            if not isascii(text):
                continue
            valid.append(text)

        return valid

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, index):

        item = self.filtered[index]
        if len(item.split()) < 2:
            rng = np.random.default_rng(index)
            new_idx = rng.choice(len(self))
            # print(f"Skipping empty element {item}, @{index} for idx {new_idx}")
            return self[new_idx]
        
        return item

def generateDataset(
    writeDir, 
    process=True,
    sample=int(1e6),
    set='train', 
    pct='10'
):

    data_loc = f"{writeDir}/data/{set}"
    if not os.path.exists(data_loc):
        os.makedirs(data_loc)

    wiki_loc = f"{writeDir}/hf"  # f"{writeDir}/wikitext"
    if not os.path.exists(wiki_loc):
        os.makedirs(wiki_loc)

    wikitext = load_dataset(
        'wikitext', 
        'wikitext-103-raw-v1', 
        cache_dir=wiki_loc, 
        split=f'{set}[:{pct}%]'
    )

    random.seed(123)
    wiki_len = len(wikitext['text']) - 100
    if wiki_len <= sample:
        passage_idxs = list(range(wiki_len))
    else:
        passage_idxs = random.sample(range(1, wiki_len), sample)
    res_list = list(itemgetter(*passage_idxs)(wikitext['text'])) 
    sampleText = filterText(res_list)
    dp = DataProcessor(sampleText, write_dir=data_loc)

    if process:
        print("running processor")
        dp.keep_ents = ['PERSON']
        dp.processEnts()
        print(dp)   
        dp.permuteEnts()
        print(dp)
    else:
        return dp


def selfSampleDataset(
    writeDir,
    sample=int(1e6),
    set='train',
    pct='10'
):
    data_loc = f"{writeDir}/data/self_sample"
    if set == 'train':
        data_loc = os.path.join(data_loc, 'train')
    elif set == 'validation':
        data_loc = os.path.join(data_loc, 'valid')
    elif set == 'test':
        data_loc = os.path.join(data_loc, 'test')
    if not os.path.exists(data_loc):
        os.makedirs(data_loc)

    wiki_loc = f"{writeDir}/hf"
    if not os.path.exists(wiki_loc):
        os.makedirs(wiki_loc)

    wikitext = load_dataset(
            'wikitext', 
            'wikitext-103-raw-v1', 
            cache_dir=wiki_loc, 
            split=f'{set}[:{pct}%]'
        )

    random.seed(123)
    wiki_len = len(wikitext['text']) - 100
    if wiki_len <= sample:
        passage_idxs = list(range(wiki_len))
    else:
        passage_idxs = random.sample(range(1, wiki_len), sample)
    res_list = list(itemgetter(*passage_idxs)(wikitext['text']))
    sampleText = filterText(res_list)

    finetuned, tokenizer = utils.loadTrainedModel(
        f"{loc}/models/finetune/gpt2_epoch0_ts10000.20210408.09.04.1617899457", 
        cache_dir=loc,
        tokenizer=True
    )
    finetuned.to(DEVICE)
    finetuned.eval()

    for idx, sent in enumerate(sampleText):

        origPath = os.path.join(data_loc, f'original_text.{idx}')
        genPath = os.path.join(data_loc, f'generated_text.{idx}')

        if os.path.exists(origPath) and os.path.exists(genPath):
            print(f'Sample {idx} already exists')
            continue

        origFile = open(origPath, 'w')
        genFile = open(genPath, 'w')

        orig_sent = sent.strip('\n').strip(" ")
        lm_tokens = torch.tensor(tokenizer(
            " "+orig_sent,
            truncation=True,
            max_length=200, 
            padding="max_length"
        )['input_ids'], device=DEVICE)
        lm_tokens = lm_tokens[lm_tokens != 50256]
        len_lm = lm_tokens.shape[-1]
        edit_loc = max(random.randint(int(len_lm*0.5), int(len_lm*0.8)), 15)
        input_ids = lm_tokens[:edit_loc]
        input_size = input_ids.size()[-1]

        print(f"generating, {DEVICE}")
        output_sequence = finetuned.generate(
            input_ids=input_ids.unsqueeze(0),
            max_length=input_size + 5,
            temperature=1.2,
            do_sample=True,
            repetition_penalty=5.0,
            num_return_sequences=10,
        )

        gen_tokens = random.choice(output_sequence)
        gen_sent = tokenizer.decode(gen_tokens, clean_up_tokenization_spaces=False)[1:]  # remove leading space

        origFile.write(orig_sent + "\n")
        genFile.write(gen_sent + "\n")

        origFile.close()
        genFile.close()


def _test(loc):
    import glob
    from transformers import GPT2Tokenizer
    
    print(f'Running from {loc}')

    data_path = f"{loc}/data/self_sample"
    writtenFiles = glob.glob(f"{data_path}/train/*")
    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))

    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2', cache_dir="/scr/em7"
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    d = NTokenDataset(list(range(fileIndex)), tokenizer, loc, n_edits=5)
    start = time.time()
    for idx in range(100):
        d[idx]
    print(time.time()-start)

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)  
    parser.add_argument('--test', action='store_true', default=False)  
    parser.add_argument('--self_sample', action='store_true', default=False)
    parser.add_argument('--run_test', action='store_true')
    args = parser.parse_args()

    loc = utils.sailPreprocess()

    if args.run_test:
        _test(loc)
        exit()
    
    func = selfSampleDataset if args.self_sample else generateDataset
    if args.train:
        print("generating training set")
        func(
            loc, 
            # process=True,
            sample=int(2e6),
            set='train', 
            pct='5'
        )
    
    if args.valid:
        print("generating eval set")
        func(
            loc, 
            sample=int(5e6), 
            set='validation', 
            pct='100'
        )

    if args.test:
        print("generating test set")
        func(
            loc, 
            sample=int(5e6), 
            set='test', 
            pct='100'
        )
