#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:35:56 2023

@author: forskningskarin

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
from torchvision import models, transforms, datasets
import torch
from supportive_code.data_setup import create_dataloaders
from torch import nn
from torchinfo import summary
from supportive_code.prediction_setup import create_predict_dataloader, predict_to_csvs, find_best_thresholds
from supportive_code.helper import show_model, create_confusion_matrix
import ast
from pathlib import Path
import torch.nn.functional
from supportive_code.padding import NewPad, NewPadAndTransform
import numpy as np
import tensorflow as tf

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (test or all)', default='test')
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='main')


    
    if parser.parse_args().data == "test":
        data_path = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Files_for_unclassifiable_test/for_building_code")

    if parser.parse_args().model == "main":
        model_path = base_dir / 'data' / 'models' /'model_without_uncl_development'
    else:
        model_path = base_dir / 'data' / 'models' / parser.parse_args().model 

    figures_path =  model_path / 'figures'
    model_save_path = model_path / 'model.pth'
    batch_size = 32

    train_transform = transforms.Compose([
                NewPadAndTransform(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness = [0.95,1.1]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])

    simple_transform = transforms.Compose([
            NewPad(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    
    with_unclassifiable = False


    _, _, _, _, BD_dataloader, class_names, class_to_idx = create_dataloaders( 
        data_path = data_path,
        transform = train_transform,
        simple_transform = simple_transform,
        batch_size = batch_size,
        with_unclassifiable = with_unclassifiable
    )
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_names)

    # load model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    ) 
    
    model.load_state_dict(torch.load(model_save_path)) # This line uses .load() to read a .pth file and load the network weights on to the architecture.
    model.to(device)
    model.eval() # enabling the eval mode to test with new samples.
   
    threshold_df = find_best_thresholds(model=model, dataloader=BD_dataloader, class_names=class_names, figures_path=figures_path)
    print(threshold_df)
    threshold_df.to_csv(model_path / 'thresholds.csv')