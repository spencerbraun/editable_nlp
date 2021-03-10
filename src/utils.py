import glob
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data_process import TorchDataset

def loadOTSModel():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def retrieveDataloader(tokenizer, bs=10, dataset='train', max_obs=float('inf')):

    writtenFiles = (
        glob.glob("../data/permuted*") if dataset == 'train' 
        else glob.glob("../data/valid/original*")
        )

    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    limitIndex = min(max_obs, fileIndex)
    dataset = TorchDataset(list(range(limitIndex)), tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=4,
        pin_memory=True
    )

    return dataloader


def loadTrainedModel(modelPath):
    model, tokenizer = loadOTSModel()
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    return model, tokenizer