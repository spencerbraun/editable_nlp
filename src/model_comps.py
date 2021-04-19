import argparse
import os
import torch

import utils
import train
import config
import evaluate as ev

from shutil import copyfile

def modelCompare(eval_path, loc):
    finetune = f'{loc}/models/finetune/gpt2_epoch0_ts10000.20210416.09.04.1618591490'

    mc = ev.ModelComps(
        model_name=eval_path
        base_name=finetune
    )
    print(mc.summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--eval_path')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--max_iter', default=20000, type=int)
    parser.add_argument('--cedit', default=1, type=float)
    parser.add_argument('--cloc', default=1, type=float)
    parser.add_argument('--lr_lr', default=1e-3, type=float)
    parser.add_argument('--edit_steps', default=1, type=int)
    parser.add_argument('--dest_path', default='/juice/scr/spencerb/editable_nlp', type=int)
    args = parser.parse_args()

    loc = utils.sailPreprocess()
    tokenizer = utils.loadTokenizer(cache_dir=loc)

    if args.compare:
        modelCompare(args.eval_path, loc)
    else:

        dataloader = utils.wikiDataloader(
                tokenizer,
                bs=args.bs,
                data_loc=loc,
                dataset='train',
                shuffle=False,
                max_length=200,
                min_length=20
            )
        
        config = config.SelfSampleConfig()
        config.write_loc = loc
        config.bs = args.bs
        config.cedit = args.cedit
        config.cloc = args.cloc
        config.lr_lr = args.lr_lr
        config.max_iter = args.max_iter
        
        trainer = train.SelfSampleTrainer(config, dataloader, tokenizer)
        trainer.run()

        hyperpath = f"{dest_path}/models/{os.path.basename(config.hyperspath)}"
        torch.save(config, hyperpath)

        ds = 'validation' 
        validation = utils.wikiDataloader(tokenizer, bs=1, data_loc=loc, dataset=ds)
        basepath = lambda step, ts: f"self_sample_epoch0_ts{step}.{ts}"

        for step in [10000,20000]:
            filename = basepath(step, trainer.timestamp)
            evalSelfSample(
                trainer.model, 
                validation,
                f"{loc}/model/{filename}", 
                int(args.edit_steps),
                loc=loc,
                testset=False
                )

            copyfile(
                f"{loc}/model/{filename}",
                f"{args.dest_path}/eval/{filename}",
            )