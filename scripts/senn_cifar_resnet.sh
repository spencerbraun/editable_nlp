#!/bin/bash
#SBATCH --partition=iris # Run on IRIS nodes
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=4 # Request 8 CPUs for this task
#SBATCH --mem=32GB # Request 64GB of memory
#SBATCH --gres=gpu:1 # Request one GPU

script=$1

source ../env/bin/activate
echo "python3 -m vision.${script} experiment=senn_cifar_resnet n_edits=$SLURM_ARRAY_TASK_ID"
python3 -m vision.${script} experiment=senn_cifar_resnet n_edits=$SLURM_ARRAY_TASK_ID

