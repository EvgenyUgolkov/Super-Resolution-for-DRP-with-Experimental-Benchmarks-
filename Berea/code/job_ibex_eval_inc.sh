#!/bin/bash
#SBATCH --job-name=Berea_recon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=36:00:00
#SBATCH --mem=250G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Evaluation.py -d paper_2_Berea_7um_5 -phases_idx 1 2 3 -sf 16 -volume_size_to_evaluate 256 256 256 -g_image_path subvolume_14.tif