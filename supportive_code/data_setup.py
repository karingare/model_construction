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
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch.utils.data as data
from PIL import Image
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
import random
import os
from PIL import Image, UnidentifiedImageError
from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        # List all directories in the given path
        class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        
        # Filter out classes that are empty or do not contain at least n_minimum non-empty valid PNG files
        n_minimum = 1
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










    


    
