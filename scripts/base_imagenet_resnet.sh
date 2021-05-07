#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=8 # Request 8 CPUs for this task
#SBATCH --mem=32GB # Request 64GB of memory
#SBATCH --gres=gpu:1 # Request one GPU

script=$1

source env/bin/activate
echo "python3 src/vision/${script}.py experiment=base_imagenet_resnet"
python3 src/vision/${script}.py experiment=base_imagenet_resnet
