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
    transform: transforms.Compose, 
    simple_transform: transforms.Compose,
    batch_size: int,
    with_unclassifiable: bool = False):

    # Load the full dataset without any transforms
    A_dataset = datasets.ImageFolder(data_path / "A", transform = transform)
    B_dataset = datasets.ImageFolder(data_path / "B", transform = simple_transform)
    CE_dataset = datasets.ImageFolder(data_path / "CE", transform = simple_transform)
    AD_dataset = datasets.ImageFolder(data_path / "AD", transform = transform)
    BD_dataset = datasets.ImageFolder(data_path / "BD", transform = simple_transform)

    num_workers = 32 # this fits with berzelius

    # define dataloaders

    A_RandSampler = RandomSampler(A_dataset, replacement=False, num_samples=None, generator=None)
    AD_RandSampler = RandomSampler(AD_dataset, replacement=False, num_samples=None, generator=None)

    A_dataloader = DataLoader(dataset=A_dataset, batch_size=batch_size, num_workers=num_workers, sampler = A_RandSampler)
    AD_dataloader = DataLoader(dataset=AD_dataset, batch_size=batch_size, num_workers=num_workers, sampler = AD_RandSampler)

    B_dataloader = DataLoader(dataset=B_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    CE_dataloader = DataLoader(dataset=CE_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    BD_dataloader = DataLoader(dataset=BD_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    if with_unclassifiable:
        classes, class_to_idx = find_classes(data_path / 'AD')
    elif not with_unclassifiable:
        classes, class_to_idx = find_classes(data_path / 'A')

    return A_dataloader, B_dataloader, CE_dataloader, AD_dataloader, BD_dataloader, classes, class_to_idx










    
