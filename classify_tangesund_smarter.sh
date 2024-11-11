#!/bin/bash
#SBATCH -A berzelius-2024-142
#SBATCH --gpus 1
#SBATCH -t 6:00:00

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate env3

model_name="tangesund_aug_16_all_classes"

# Extract directories
ls /proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/*.tar | xargs -n 1 -P 8 tar -x -C /scratch/local/ -f

# train the model on the tangesund data
cd /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction
python train.py --data=tangesund --model_folder_name=$model_name --num_epochs=20
python determine_thresholds.py --data=tangesund --model=$model_name
python test_on_test_data.py --data=tangesund --model=$model_name
python sample_model_predictions.py --data=tangesund --model=$model_name
##python sample_model_predictions.py --data=tangesund_6_m --model=tangesund_aug_14_1123 --sample_size=100000


# for 16 m
echo "Working on 16 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_16"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 13 m
echo "Working on 13 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_13_png"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 11 m
echo "Working on 11 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_11_png"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 8 m
echo "Working on 8 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_8_png"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 6 m
echo "Working on 6 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_6_png"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name

# for 3 m
echo "Working on 3 m predictions"
image_data_path="/scratch/local/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/original_files/raw_3_png"
find $image_data_path -type f -size 0 -delete
python /proj/berzelius-2023-48/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# Define the model folder path
model_folder="/proj/berzelius-2023-48/ifcb/main_folder_karin/data/models/${model_name}"

# Create directory to store CSV files
mkdir -p /scratch/local/csv_files

# Initialize a counter for labeling files
counter=1

# Find and copy all image_class_table.csv files to the csv_files directory with sequential numbering
find "$model_folder" -name "image_class_table.csv" | while read -r file; do
    cp "$file" "/scratch/local/csv_files/image_class_table_${counter}.csv"
    ((counter++))
done

# Directory containing your CSV files
input_dir="/scratch/local/csv_files"

# Output file
output_file="/proj/berzelius-2023-48/ifcb/main_folder_karin/data/models/${model_name}/image_class_table_combined.csv"

# Temporary file for processing
temp_file="/proj/berzelius-2023-48/ifcb/main_folder_karin/data/models/${model_name}/image_class_table_combined_temp.csv"

# Check if there are any CSV files to process
if ls "$input_dir"/*.csv 1> /dev/null 2>&1; then
    # Combine the files, retaining the header only from the first file
    awk 'FNR==1 && NR!=1 { next; } { print }' "$input_dir"/*.csv > "$temp_file"
    
    # Reset row numbers in the combined CSV
    awk 'BEGIN { FS=OFS="," } NR==1 { print; next } { $1=NR-1; print }' "$temp_file" > "$output_file"
    
    # Remove the temporary file
    rm "$temp_file"
    
    echo "DONE :)"
else
    echo "No CSV files found in $input_dir"
fi