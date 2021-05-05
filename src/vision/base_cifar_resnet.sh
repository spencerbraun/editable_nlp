#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=8 # Request 8 CPUs for this task
#SBATCH --mem=32G # Request 64GB of memory
#SBATCH --gres=gpu:1 # Request one GPU

source /iris/u/clin/code/editable_nlp/env/bin/activate
python3 src/vision/train.py experiment=base_cifar_resnet
