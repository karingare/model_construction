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
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
import random


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
    batch_size: int
):
    """Creates training, testing and validation DataLoaders.

    Takes in a directory path with separate folders "train", "test"  etc and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: torchvision transforms to perform on training and testing data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for number of workers per DataLoader.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, test_dataloader, class_names = \
          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               transform=some_transform,
                               batch_size=32,
                               num_workers=4)
    """



  # Create an image folder class. This class is used to load the images from the directory, and transform them later. Typically, the images are transofrmed during laoding
      

      
    # Create a transform for the training data    
    # Load the full dataset without any transforms
    full_dataset = datasets.ImageFolder(data_path)
    unclassifiable_dataset = datasets.ImageFolder(unclassifiable_path)


    # Filter out empty classes
    class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    class_names_filtered = []
    class_to_idx_filtered = {}
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        if len(os.listdir(class_path)) > 0:  # Check if class folder is not empty
            class_names_filtered.append(class_name)
            class_to_idx_filtered[class_name] = class_to_idx[class_name]


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

    test_with_unclassifiable_dataset = ConcatDataset([test_dataset, unclassifiable_dataset])
    val_with_unclassifiable_dataset = ConcatDataset([val_dataset, unclassifiable_dataset])



    RandSampler = RandomSampler(train_dataset, replacement=False, num_samples=None, generator=None)

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

    classes, class_to_idx = find_classes(data_path)


    return train_dataloader, val_dataloader, val_with_unclassifiable_dataloader, test_dataloader, test_with_unclassifiable_dataloader, classes, class_to_idx










    
