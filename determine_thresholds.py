#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:35:56 2023

@author: forskningskarin

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
from torchvision import models, transforms
import torch
from supportive_code.data_setup import create_dataloaders
from torch import nn
from supportive_code.prediction_setup import find_best_thresholds
from pathlib import Path
from supportive_code.padding import NewPad, NewPadAndTransform

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_dir = Path("/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (test or all)', default='test')
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='development')

    print(parser.parse_args().data)
    if parser.parse_args().data == "test":
        data_path = base_dir / 'data' /'development'
        unclassifiable_path = base_dir / 'data' / 'development_unclassifiable'
    elif parser.parse_args().data == "smhibaltic2023":
        data_path = base_dir / 'data' / 'smhi_training_data_oct_2023' / 'Baltic'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "syke2022":
        data_path = Path('/proj/common-datasets/SYKE-plankton_IFCB_2022/20220201/phytoplankton_labeled/labeled_20201020')
        unclassifiable_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "tangesund":
        data_path = '/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SMHI_IFCB_Plankton_Image_Reference_Library_v4/smhi_ifcb_tangesund_annotated_images'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "tangesund_skagerrak_kattegat_merged":
        data_path = '/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/SMHI_IFCB_Plankton_tangesund_and_skagerrak_v4'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "amime":
        data_path = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/AMIME_main_dataset"
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'

    if parser.parse_args().model == "main":
        model_path = base_dir / 'data' / 'models' /'model_main_240116' 
    elif parser.parse_args().model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209' 
    elif parser.parse_args().model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
    else:
        model_path = base_dir / 'data' / 'models' / parser.parse_args().model 


    print("[INFO]: Determining thresholds")

    training_info_path = model_path / 'training_info.txt'

    # Read the file contents
    training_info = {}
    with open(training_info_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            # Try to evaluate value if it's a list or int/float, otherwise keep it as a string
            try:
                value = eval(value)
            except (SyntaxError, NameError):
                pass
            training_info[key] = value

    # Example: Access the padding_mode
    padding_mode = training_info.get('padding_mode')

    figures_path =  model_path / 'figures'
    model_save_path = model_path / 'model.pth'
    batch_size = 32

    train_transform = transforms.Compose([
                NewPadAndTransform(padding_mode=padding_mode),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness = [0.95,1.1]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])

    simple_transform = transforms.Compose([
            NewPad(padding_mode=padding_mode),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    


    # create dataset and dataloader for the valid dataset
    train_dataloader, val_dataloader, val_with_unclassifiable_dataloader, test_dataloader, test_with_unclassifiable_dataloader, class_names, class_to_idx = create_dataloaders( 
        data_path = data_path,
        unclassifiable_path = unclassifiable_path,
        transform = train_transform,
        simple_transform = simple_transform,
        batch_size = batch_size
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
   
    threshold_df = find_best_thresholds(model=model, dataloader=val_with_unclassifiable_dataloader, class_names=class_names, figures_path=figures_path)
    print(threshold_df)
    threshold_df.to_csv(model_path / 'thresholds.csv')