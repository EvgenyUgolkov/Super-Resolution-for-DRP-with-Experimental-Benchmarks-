#!/bin/bash
#SBATCH --job-name=Pe6i3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=71:00:00
#SBATCH --mem=350G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Evaluation.py -d paper_3_x16_Parker_1_long_6 -phases_idx 1 2 3 -sf 16 -volume_size_to_evaluate 256 256 256 -g_image_path Parker_iteration_3.tif