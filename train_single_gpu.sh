#!/bin/sh
#SBATCH --job-name=train_sheepherding # job name
#SBATCH --output=train_output_sbatch.txt # file name to redirect the output
#SBATCH --time=12:00:00 # job time limit - full format is D-H:M:S
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --ntasks=1 # number of tasks
#SBATCH --partition=gpu # partition to run on nodes that contain gpus
#SBATCH --mem-per-gpu=16G # memory allocation

source ~/miniconda3/etc/profile.d/conda.sh # intialize conda
conda activate pytorch_env # activate the previously created environment
srun python train.py -n initial --cuda 
#srun python sheepherding.py 