#!/bin/bash
#SBATCH --job-name=Parker_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=02:00:00
#SBATCH --mem=450G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Architecture.py -d paper_3_x16_Parker_1_long_6 --with_rotation -phases_idx 1 2 3 -sf 16 -g_image_path Parker_r.tif -d_image_path Parker_M_0.tif Parker_M_1.tif Parker_M_2.tif Parker_M_3.tif Parker_HR.tif