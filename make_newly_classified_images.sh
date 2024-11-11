#!/bin/bash
#SBATCH -A berzelius-2024-142
#SBATCH --gpus 1
#SBATCH -t 3:00:00


module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate env3

model_name="tangesund_aug_16_all_classes_turned"

# Extract directories
ls /proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/*.tar | xargs -n 1 -P 8 tar -x -C /scratch/local/ -f

# train the model on the tangesund data
cd /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction
python train.py --data=tangesund --model_folder_name=$model_name --num_epochs=30
python determine_thresholds.py --data=tangesund --model=$model_name
python test_on_test_data.py --data=tangesund --model=$model_name
python sample_model_predictions.py --data=tangesund_3_m --model=$model_name --sample_size=10000000