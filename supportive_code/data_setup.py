#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:34:00 2023

@author: forskningskarin

Contains functions for creating dataloaders
"""

import os
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import io
import glob
from functools import partial
from torchvision import datasets, transforms
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import Subset
from torchvision import datasets
from torch.utils.data import DataLoader,  RandomSampler
from torchdata.datapipes.iter import FileLister, FileOpener

from PIL import Image
from pathlib import Path
import os
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.transforms as transforms

class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # List all directories in the given path
        class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        
        # Filter out classes that are empty or do not contain at least n_minimum non-empty valid PNG files
        n_minimum = 50  # Minimum number of valid images per class
        class_names_filtered = []
        class_to_idx_filtered = {}
        
        for class_name in class_names:
            class_path = os.path.join(directory, class_name)
            # List all files in the directory
            files = os.listdir(class_path)
            valid_png_files = []
            
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(class_path, file)
                    if os.path.getsize(file_path) > 0:
                        try:
                            img = Image.open(file_path)
                            img.verify()  # Verify that it is, indeed, an image
                            valid_png_files.append(file)
                        except (IOError, SyntaxError) as e:
                            print(f'Invalid image file: {file_path} - {e}')

            if len(valid_png_files) >= n_minimum:  # Check if class folder contains at least n_minumum valid PNG files
                class_names_filtered.append(class_name)
                class_to_idx_filtered[class_name] = class_to_idx[class_name]

        class_names_filtered.sort()
        class_to_idx_filtered = {class_name: i for i, class_name in enumerate(class_names_filtered)}
        
        return class_names_filtered, class_to_idx_filtered



class CustomImageFolderOptimizedOld(datasets.ImageFolder):
    def find_classes(self, directory):
        # List all directories in the given path
        class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        
        n_minimum = 1  # Minimum number of valid images per class

        def is_valid_png(file_path):
            if os.path.getsize(file_path) > 0:
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify that it is, indeed, an image
                    return True
                except (IOError, SyntaxError):
                    return False
            return False

        def process_class(class_name):
            class_path = os.path.join(directory, class_name)
            files = os.listdir(class_path)
            valid_files = [
                file for file in files if file.endswith('.png') and is_valid_png(os.path.join(class_path, file))
            ]
            return class_name if len(valid_files) >= n_minimum else None

        # Use multithreading to speed up file processing
        class_names_filtered = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_class, class_name): class_name for class_name in class_names}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    class_names_filtered.append(result)

        # Sort filtered class names and create a class-to-index mapping
        class_names_filtered.sort()
        class_to_idx_filtered = {class_name: i for i, class_name in enumerate(class_names_filtered)}

        return class_names_filtered, class_to_idx_filtered



# a test for prediction 
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, path
        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️ Skipping unreadable image: {path} ({e})")
            return None  # Will be filtered out by custom collate_fn


import tarfile

def is_valid_tar(filepath):
    try:
        with tarfile.open(filepath, 'r') as tar:
            tar.getmembers()
        return True
    except tarfile.TarError:
        return False
 
class ImageWebDataset(IterableDataset):
    def __init__(self, paths, transform=None):
        # Normalize input to list of paths
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.transform = transform

        # Recursively find all .tar files in the supplied paths
        tar_files = []
        for path in self.paths:
            if os.path.isdir(path):
                found = glob.glob(os.path.join(path, "**", "*.tar"), recursive=True)
                tar_files.extend(found)
            else:
                print(f"⚠️ Skipping non-directory: {path}")
        
        if not tar_files:
            raise FileNotFoundError(f"No .tar files found in: {self.paths}")

        tar_files = [f for f in tar_files if is_valid_tar(f)]

        self.datapipe1 = iter(tar_files)
        self.datapipe2 = FileOpener(self.datapipe1, mode="b")

    def __iter__(self):
        dataset = self.datapipe2.load_from_tar().map(partial(decode, transform=self.transform)).filter(lambda x: x is not None)
        for obj in dataset:
            yield obj

def decode(item, transform=None):
    key, value = item
    if not key.endswith(".png"):
        return None
    try:
        img_bytes = value.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if transform:
            img = transform(img)
        return img, key
    except Exception as e:
        print(f"⚠️ Failed to decode {key}: {e}")
        return None

    

class CustomImageFolderOptimized(datasets.ImageFolder):
    def find_classes(self, directory):
        class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        return sorted(class_names), {class_name: i for i, class_name in enumerate(sorted(class_names))}

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None, allow_empty=False):
        def is_valid_png(file_path):
            if os.path.getsize(file_path) > 0:
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    return True
                except Exception:
                    return False
            return False

        # Based on torchvision.datasets.folder.make_dataset
        instances = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if fname.lower().endswith(".png") and is_valid_png(path):
                        item = (path, class_index)
                        instances.append(item)

        return instances




class CustomImageFolderAllImages(datasets.ImageFolder): 
    def find_classes(self, directory):
        # List all directories in the given path
        class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        
        # Filter out classes that are empty or do not contain at least 30 non-empty valid PNG files
        class_names_filtered = []
        class_to_idx_filtered = {}
        
        for class_name in class_names:
            class_path = os.path.join(directory, class_name)
            # List all files in the directory
            files = os.listdir(class_path)
            valid_png_files = []
            
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(class_path, file)
                    if os.path.getsize(file_path) > 0:
                        try:
                            img = Image.open(file_path)
                            img.verify()  # Verify that it is, indeed, an image
                            valid_png_files.append(file)
                        except (IOError, SyntaxError) as e:
                            print(f'Invalid image file: {file_path} - {e}')

            if len(valid_png_files) >= 1:  # Check if class folder contains at least 30 valid PNG files
                class_names_filtered.append(class_name)
                class_to_idx_filtered[class_name] = class_to_idx[class_name]

        class_names_filtered.sort()
        class_to_idx_filtered = {class_name: i for i, class_name in enumerate(class_names_filtered)}
        
        return class_names_filtered, class_to_idx_filtered
        

def find_classes(dir):
    classes = os.listdir(dir)
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def create_dataloaders(
    data_path: Path,
    unclassifiable_path: Path,
    transform: transforms.Compose, 
    simple_transform: transforms.Compose,
    batch_size: int,
    filenames: bool = False, 
    model_path: str = None,
    boost_dataset: str = None
):
    # Create a transform for the training data    
    # Load the full dataset without any transforms

    unclassifiable_dataset = CustomImageFolder(unclassifiable_path)

    # if boost_dataset is not none
    if boost_dataset is not None:
        boost_dataset = CustomImageFolder(boost_dataset)
        full_dataset = CustomImageFolder(data_path)

        classes = []
        class_to_idx = {}
    
        for dataset in [boost_dataset, full_dataset]:
            if hasattr(dataset, 'classes') and hasattr(dataset, 'class_to_idx'):
                classes.extend(dataset.classes)
                class_to_idx.update(dataset.class_to_idx)

        # Ensure unique class names and indices
        classes = list(set(classes))  # Remove duplicates
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            # Ensure unique class names and indices
        classes = list(set(classes))  # Remove duplicates
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        full_dataset = ConcatDataset([full_dataset, boost_dataset])
    else:
        full_dataset = CustomImageFolder(data_path)
        classes, class_to_idx = full_dataset.classes, full_dataset.class_to_idx

    # Split the dataset into training and test sets
    train_indices, test_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

    # Split the training set into training and validation sets
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)

    # Split the unclassifiable images into two groups
    unclassifiable_val_indices, unclassifiable_test_indices = train_test_split(list(range(len(unclassifiable_dataset))), test_size=0.5, random_state=42)

    # Make the unclassifiable dataset at least as large as the test dataset
    assert len(unclassifiable_val_indices) >= len(val_indices), f"The unclassifiable_dataset is not large enough, {len(unclassifiable_val_indices)+ len(unclassifiable_test_indices)} images. Should be at least {len(val_indices)+ len(test_indices)} images."
    assert len(unclassifiable_test_indices) >= len(test_indices), f"The unclassifiable_dataset is not large enough, {len(unclassifiable_val_indices)+ len(unclassifiable_test_indices)} images. Should be at least {len(val_indices)+ len(test_indices)} images."

    num_test_images = len(test_indices)
    num_val_images = len(val_indices)

    # Adjust the size of unclassifiable_test_indices to match the number of test images
    unclassifiable_test_indices = unclassifiable_test_indices[:num_test_images]
    unclassifiable_val_indices = unclassifiable_val_indices[:num_val_images]

    # Create Dataset objects for each subset with the appropriate transforms
    train_dataset = Subset(full_dataset, train_indices)
    train_dataset.dataset.transform = transform

    val_dataset = Subset(full_dataset, val_indices)
    val_dataset.dataset.transform = simple_transform

    test_dataset = Subset(full_dataset, test_indices)
    test_dataset.dataset.transform = simple_transform

    unclassifiable_val_dataset = Subset(unclassifiable_dataset, unclassifiable_val_indices)
    unclassifiable_val_dataset.dataset.transform = simple_transform

    unclassifiable_test_dataset = Subset(unclassifiable_dataset, unclassifiable_test_indices)
    unclassifiable_test_dataset.dataset.transform = simple_transform

    test_with_unclassifiable_dataset = ConcatDataset([test_dataset, unclassifiable_test_dataset])
    val_with_unclassifiable_dataset = ConcatDataset([val_dataset, unclassifiable_val_dataset])

    RandSampler = RandomSampler(train_dataset, replacement=False, num_samples=None, generator=None)
  
    if filenames:  # save a file with the names of the training, testing and validation images
        with open(model_path / 'train_filenames.txt', 'w') as f:
            for idx in train_indices:
                f.write(os.path.basename(full_dataset.samples[idx][0] + '\n'))
        with open(model_path / 'val_filenames.txt', 'w') as f:
            for idx in val_indices:
                f.write(os.path.basename(full_dataset.samples[idx][0] + '\n'))
        with open(model_path / 'test_filenames.txt', 'w') as f:
            for idx in test_indices:
                f.write(os.path.basename(full_dataset.samples[idx][0] + '\n'))

	  # define dataloaders
    num_workers = 32 # this fits with berzelius
    train_dataloader = DataLoader(dataset=train_dataset, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers,
                                     sampler = RandSampler)
    

    val_dataloader = DataLoader(dataset=val_dataset,
		                             batch_size=batch_size, 
		                             num_workers=num_workers, 
		                             shuffle=False) # don't usually need to shuffle testing data
    

    val_with_unclassifiable_dataloader = DataLoader(dataset=val_with_unclassifiable_dataset,
		                             batch_size=batch_size, 
		                             num_workers=num_workers, 
		                             shuffle=True) # don't usually need to shuffle testing data
    
    test_dataloader = DataLoader(dataset=test_dataset,
		                             batch_size=batch_size, 
		                             num_workers=num_workers, 
		                             shuffle=False) 
    
    test_with_unclassifiable_dataloader = DataLoader(dataset=test_with_unclassifiable_dataset,
		                             batch_size=batch_size, 
		                             num_workers=num_workers, 
		                             shuffle=False) 

    return train_dataloader, val_dataloader, val_with_unclassifiable_dataloader, test_dataloader, test_with_unclassifiable_dataloader, classes, class_to_idx










    


    
