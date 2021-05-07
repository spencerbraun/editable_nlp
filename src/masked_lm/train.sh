#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=16G # Request 64GB of memory
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --exclude=jagupard4,jagupard5,jagupard6,jagupard7,jagupard8,jagupard12,jagupard14,jagupard16,jagupard17,jagupard18,jagupard19,jagupard20,jagupard21,jagupard22,jagupard23,jagupard24,jagupard25,jagupard26,jagupard27,jagupard28,jagupard29
#SBATCH --job-name="t5_finetune"
eval "$(conda shell.bash hook)" # Needed to get conda to work in the script for some reason, see this comment: https://github.com/conda/conda/issues/7980#issuecomment-492784093
conda activate editable
python /juice/scr/spencerb/editable_nlp/src/masked_lm/train.py  
