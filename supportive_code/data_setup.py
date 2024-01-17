#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:34:00 2023

@author: forskningskarin

Contains functions for creating dataloaders
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torch.utils.data as data
import numpy as np
from pathlib import Path
import random


def find_classes(dir):
    classes = os.listdir(dir)
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def create_dataloaders(
    split_data_path: Path,
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


	# dataset creation with ImageFolder
    train_path = split_data_path / "train"
    val_path = split_data_path / "val"
    val_with_unclassifiable_path = split_data_path / "valWithUnclassifiable"
    test_path = split_data_path / "test"
    test_with_unclassifiable_path = split_data_path / "testWithUnclassifiable"

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=simple_transform)
    val_with_unclassifiable_dataset = datasets.ImageFolder(root=val_with_unclassifiable_path, transform=simple_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=simple_transform)
    test_with_unclassifiable_dataset = datasets.ImageFolder(root=test_with_unclassifiable_path, transform=simple_transform)

    # create sampler for the train dataset
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

    classes, class_to_idx = find_classes(train_path)


    return train_dataloader, val_dataloader, val_with_unclassifiable_dataloader, test_dataloader, test_with_unclassifiable_dataloader, classes, class_to_idx










    
