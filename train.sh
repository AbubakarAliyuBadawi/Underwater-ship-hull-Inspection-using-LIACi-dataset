#!/bin/bash
#SBATCH --job-name=img_seg           # Job name
#SBATCH --output=LIACi-04-closed-world_%j.log         # Output file
#SBATCH --error=LIACi-04-closed-world_%j.log           # Error file
#SBATCH --partition=GPUQ               # Partition name for GPU access
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU with 40GB
#SBATCH --cpus-per-task=4              # Request 4 CPU cores
#SBATCH --mem=40G                      # Request 40 GB of RAM
#SBATCH --time=10:00:00                # Set a time limit of 1 hour

# Load any necessary modules or activate your conda environment
module load CUDA/12.1.1
source ~/miniconda3/bin/activate openworld

# Run your Python script
python /cluster/home/abubakb/ContMAV_3/train.py --id "LIACi-03-closed-world-02" --dataset_dir "/cluster/home/abubakb/final_dataset" --num_classes 11 --batch_size 4
