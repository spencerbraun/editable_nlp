import argparse
import os
import torch

import utils
from evaluate import ModelComps

def main(model_name):

    comps = ModelComps(
        model_name=model_name,
        base_name="gpt2_epoch0_ts10000.20210310.18.03.1615401990",
        archive=False
    )

    torch.save(comps, f"../eval/comps/{model_name}.comps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='')
    args = parser.parse_args()
    
    if not os.path.exists("../eval/comps/"):
        os.mkdir("../eval/comps/")
    
    main()