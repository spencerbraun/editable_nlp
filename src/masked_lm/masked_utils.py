import glob
import os
import platform
import zipfile
from shutil import copyfile

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def loadT5Model(cache_dir=None):
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-small", cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )
    tokenizer = T5Tokenizer.from_pretrained(
        't5-small', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )

    return model, tokenizer

def loadT5Tokenizer(cache_dir=None):
    tokenizer = T5Tokenizer.from_pretrained(
        't5-small', cache_dir=f"{cache_dir}/hf" if cache_dir else None
        )

    return tokenizer

def loadTrainedT5Model(modelPath, cache_dir=None, tokenizer=True):
    model, tok = loadT5Model(cache_dir=cache_dir)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    if not tokenizer:
        return model
    return model, tok


def sailPreprocess():
    user = os.environ["USER"]
    if user == "spencerb":
        machine_name = platform.node().split(".")[0]
        scr = max(os.listdir(f"/{machine_name}"))
        save_loc = f"/{machine_name}/{scr}"
        local_dir = f"{save_loc}/{user}"

    else:
        save_loc = "/iris/u"
        local_dir = f"{save_loc}/{user}/code/editable_nlp"

    return local_dir
