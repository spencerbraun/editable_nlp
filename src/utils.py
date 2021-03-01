from transformers import GPT2Tokenizer, GPT2LMHeadModel

def loadModel():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def retrieveDataloader(set='train'):

    writtenFiles = glob.glob("../data/valid/permuted*")
    fileIndex = max(map(lambda x: int(x.split(".")[-1]), writtenFiles))
    dataset = TorchDataset(list(range(fileIndex)), tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs
    )

    return dataloader

