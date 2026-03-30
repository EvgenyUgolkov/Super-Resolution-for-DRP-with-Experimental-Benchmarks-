#!/bin/bash
#SBATCH --job-name=test_37_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=250G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Evaluation.py -d paper_3_x32_test_37 -phases_idx 1 2 3 -sf 32 -volume_size_to_evaluate 128 128 128 -g_image_path Kent_rec_2.tif