import os
import random
import shutil

# Define the source and target directories
source_dir = '/proj/common-datasets/SMHI-IFCB-Plankton/version-2/smhi_ifcb_tångesund_annotated_images'
target_dir = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_sampled_images'


# Set the number of images to sample per class
num_images_per_class = 10

# Dictionary to store images by class
images_by_class = {}

# Walk through all subdirectories and group images by class (assuming subfolder name is the class)
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(('jpg', 'png', 'jpeg', 'tif', 'bmp')):  # Add other image formats as needed
            class_name = os.path.basename(root)  # Assuming the folder name is the class name
            img_path = os.path.join(root, file)
            
            if class_name not in images_by_class:
                images_by_class[class_name] = []
            images_by_class[class_name].append(img_path)

# Sample images and copy them to the target directory
for class_name, img_paths in images_by_class.items():
    if len(img_paths) > num_images_per_class:
        sampled_images = random.sample(img_paths, num_images_per_class)
    else:
        sampled_images = img_paths  # If there are less than or equal to 10 images, take all

    for img_path in sampled_images:
        # Get the relative path from the source_dir
        relative_path = os.path.relpath(img_path, source_dir)
        
        # Create the same folder structure in the target_dir
        new_img_path = os.path.join(target_dir, relative_path)
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        
        # Copy the image to the new location
        shutil.copy2(img_path, new_img_path)

print(f"Sampled {num_images_per_class} images from each class and copied them to {target_dir}")