

# ml rocm/5.7.0
#ml craype-accel-amd-gfx90a

# ml PDC/23.12
# ml anaconda3/2024.02-1-cpeGNU-23.12
# conda activate /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/env3
# conda list

model_name="tangesund_dec_1_all_classes"


# train the model on the tangesund data
# echo "Training the model on the tangesund data"
# cd /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/model_construction
# python train.py --data=tangesund --model_folder_name=$model_name --num_epochs=20

echo "Determining thresholds"
python determine_thresholds.py --data=tangesund --model=$model_name

echo "Testing the model on the test data"
python test_on_test_data.py --data=tangesund --model=$model_name

echo "Sampling model predictions"
python sample_model_predictions.py --data=tangesund --model=$model_name --sample_size=100


# for 16 m
echo "Working on 16 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/16_m"
find $image_data_path -type f -size 0 -delete
python /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 13 m
echo "Working on 13 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/13_m"
find $image_data_path -type f -size 0 -delete
python /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/model_construction/predict.py --data="$image_data_path" --model=$model_name


# for 11 m
echo "Working on 11 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/11_m"
find $image_data_path -type f -size 0 -delete
python predict.py --data="$image_data_path" --model=$model_name


# for 8 m
echo "Working on 8 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/8_m"
find $image_data_path -type f -size 0 -delete
python predict.py --data="$image_data_path" --model=$model_name


# for 6 m
echo "Working on 6 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/6_m"
find $image_data_path -type f -size 0 -delete
python predict.py --data="$image_data_path" --model=$model_name

# for 3 m
echo "Working on 3 m predictions"
image_data_path="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/tangesund_png/6_m"
find $image_data_path -type f -size 0 -delete
python predict.py --data="$image_data_path" --model=$model_name


# Define the model folder path
model_folder="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}"

# Create directory to store CSV files
mkdir -p /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/csv_files

# Initialize a counter for labeling files
counter=1

# Find and copy all image_class_table.csv files to the csv_files directory with sequential numbering
find "$model_folder" -name "image_class_table.csv" | while read -r file; do
    cp "$file" "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/csv_files/image_class_table_${counter}.csv"
    ((counter++))
done

# Directory containing your CSV files
input_dir="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/csv_files"

# Output file
output_file="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/image_class_table_combined.csv"

# Temporary file for processing
temp_file="/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/image_class_table_combined_temp.csv"

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

# remove the csv_files directory
rm -r /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/${model_name}/csv_files