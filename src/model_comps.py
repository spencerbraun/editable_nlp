import argparse
import os
import torch

import utils

def stripPadding(tok):
    flat = tok[0].squeeze()
    return flat[flat != 50256]

def decode(tok, tokenizer):
    strp = stripPadding(tok)
    return tokenizer.decode(strp)

def main(loc, tokenizer, shuffle=True):
    
    dl = utils.retrieveDataloader(
        tokenizer, 
        bs=1, 
        data_loc=loc,
        dataset='train', 
        max_obs=float('inf'),
        shuffle=shuffle
    )

    for raw, perm, new_ent_tok, old_ent_tok in dataloader:
        for tok in [raw, perm, old_ent_tok, new_ent_tok]:
            print(decode(tok, tokenizer), '\n')
            
        val = input("Enter to continue\n\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='')
    args = parser.parse_args()

    loc = utils.sailPreprocess(debug=True)
    tokenizer = utils.loadTokenizer(cache_dir=loc)
    main(loc, tokenizer)