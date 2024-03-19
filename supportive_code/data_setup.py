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
    batch_size: int,
):
    # Load the full dataset without any transforms
    full_dataset = datasets.ImageFolder(data_path)
    unclassifiable_dataset = datasets.ImageFolder(unclassifiable_path)

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

    unclassifiable_test_dataset = Subset(unclassifiable_dataset, unclassifiable_test_indices)
    unclassifiable_test_dataset.dataset.transform = simple_transform

    unclassifiable_val_dataset = Subset(unclassifiable_dataset, unclassifiable_val_indices)
    unclassifiable_val_dataset.dataset.transform = simple_transform

    # Concatenate datasets, adding unclassifiable as the fifth class
    test_with_unclassifiable_dataset = ConcatDataset([test_dataset, unclassifiable_test_dataset])
    val_with_unclassifiable_dataset = ConcatDataset([val_dataset, unclassifiable_val_dataset])
    train_with_unclassifiable_dataset = ConcatDataset([train_dataset, unclassifiable_val_dataset])

	  
    num_workers = 32 # this fits with berzelius

    # define dataloaders

    RandSampler = RandomSampler(train_with_unclassifiable_dataset, replacement=False, num_samples=None, generator=None)
    train_dataloader = DataLoader(dataset=train_with_unclassifiable_dataset, batch_size=batch_size, num_workers=num_workers, sampler = RandSampler)

    val_dataloader = DataLoader(dataset=val_dataset,batch_size=batch_size, num_workers=num_workers, shuffle=False) # don't usually need to shuffle testing data
    val_with_unclassifiable_dataloader = DataLoader(dataset=val_with_unclassifiable_dataset,batch_size=batch_size,num_workers=num_workers, shuffle=True) # don't usually need to shuffle testing data
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










    
