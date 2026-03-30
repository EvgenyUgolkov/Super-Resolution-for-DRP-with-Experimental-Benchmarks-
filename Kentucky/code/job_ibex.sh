#!/bin/bash
#SBATCH --job-name=Kentucky_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=a100:2
#SBATCH --time=02:00:00
#SBATCH --mem=450G

echo "running pytorch"

export PYTHONUNBUFFERED=TRUE
python3 Architecture.py -d paper_3_x32_test_37 --with_rotation -phases_idx 1 2 3 -sf 32 -g_image_path Kentucky_segm_range_2.tif -d_image_path Kent_M_0_s.tif Kent_M_1_s.tif Kent_M_2_s.tif Kent_M_3_s.tif Kent_M_4_s.tif Kent_output_stack_filtered_s.tif