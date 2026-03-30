#!/bin/bash
#SBATCH --job-name=Berea_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=02:00:00
#SBATCH --mem=350G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Architecture.py -d paper_2_Berea_7um_5 --with_rotation -phases_idx 1 2 3 -sf 16 -g_image_path Berea_7um_5.tif -d_image_path LR_M_-1.tif LR_M_0.tif LR_M_1.tif LR_M_2.tif Berea_CSLM_clay_gen.tif