import collections
import random
import time
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F

import spacy
from datasets import load_dataset, list_metrics, load_metric



def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def filterText(iterator):

    valid  = []
    for text in iterator:
        if len(text) < 50:
            continue
        if not is_ascii(text):
            continue
        valid.append(text)

    return valid



class DataProcessor:
    def __init__(
        self, 
        text, 
        write_dir=None, 
        parallel=False
    ):
        self.text = text

        self.write_dir = write_dir
        self.parallel = parallel
        
        self.raw_texts = []
        self.ner_texts = []
        self.permuted = []
        self.changed_ents = []
        
        self.ents = collections.defaultdict(list)

        self.model = spacy.load(
            "en_core_web_sm", 
            exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        )
        self.model.add_pipe('sentencizer')
        
        self.keep_ents = ['PERSON', 'ORG', 'GPE']
        
    @classmethod
    def fromFile(cls, file_loc):
        pass
    
    
    def run(self, func, args):
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                for output in executor.map(func, args):
                    return output.result(timeout=None)
        else:
            for output in func(*args):
                yield output
            
    
    
    def permuteEnts(self):
        timestamp = time.time()
        
            
        for idx, (sent, ents) in enumerate(self.ner_texts):
            
            if self.write_dir:
                permuteFile = open(self.write_dir + f'/permuted_entities.{idx}', 'w')
                origFile = open(self.write_dir + f'/original_entities.{idx}', 'w')
                entFile = open(self.write_dir + f'/entity_swaps.{idx}', 'w')

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

            if self.write_dir:
                permuteFile.write(new_sent + "\n")
                origFile.write(self.raw_texts[idx].strip('\n').strip(" ") + "\n")
                entFile.write(f"{orig_ent[0]}|{replace_ent}\n")

                permuteFile.close()
                origFile.close()
                entFile.close()
                
            self.permuted.append(new_sent)
            self.changed_ents.append((orig_ent[0], replace_ent))
            
    
    
    def processEnts(self):
                
        for output in self.runNER(self.text):
            self.ner_texts.append(output)
        
        
    def runNER(self, texts):
        for doc in self.model.pipe(texts):
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


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, tokenizer, max_length=1024):
        'Initialization'
        self.list_IDs = list_IDs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        
    def tokenize(self, textList):
        tokList = []
        for idx in range(len(textList)):
            tok = self.tokenizer(
                self.tokenizer.bos_token + 
                textList[idx] + 
                self.tokenizer.eos_token,
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
        return tokList
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        with open(f"data/original_entities.{ID}") as raw:
            raw_sample = raw.read()
        with open(f"data/permuted_entities.{ID}") as perm:
            permuted_sample = perm.read()
        
        raw, perm = self.tokenize([raw_sample, permuted_sample])

        return raw, perm


def generateDataset(writeDir, set=train, pct=10)
    wikitext = load_dataset(
            'wikitext', 
            'wikitext-103-raw-v1', 
            cache_dir="/Volumes/External HD/Dev/datasets/wikitext", 
            split=f'{set}[:{pct}%]'
        )

    random.seed(123)
    passage_idxs = random.sample(range(1, 1e6), 60000)
    sampleText = filterText(wikitext['text'][passage_idxs])
    
    dp = DataProcessor(sampleText, write_dir=writeDir)
    dp.keep_ents = ['PERSON']
    dp.processEnts()
    dp.permuteEnts()

if __name__ == "__main__":

    