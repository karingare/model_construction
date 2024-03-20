#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 15:35:56 2023

@author: forskningskarin

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # decides how much info to show from tensorflow
from torchvision import models, transforms, datasets
import torch
from torch import nn
from pathlib import Path
import argparse
from supportive_code.prediction_setup import evaluate_on_test
import ast
import pandas as pd
from supportive_code.padding import NewPad
from supportive_code.data_setup import create_dataloaders

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")
    figures_path =  base_dir / 'out'

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='main_test')
    parser.add_argument('--testtype', type=str, help='Specify the type of test data to use (fraction of full set or separate set)', default='fraction')
    parser.add_argument('--data', type=str, help='Specify any specific dataset to use', default='test')

    with_unclassifiable = True
    
    if parser.parse_args().model == "main_test" and not with_unclassifiable:
        model_path = base_dir / 'data' / 'models' /'model_without_uncl_development'
    elif parser.parse_args().model == "main_test" and with_unclassifiable:
        model_path = base_dir / 'data' / 'models' /'model_with_uncl_development'

    if parser.parse_args().data == "test":
        data_path = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Files_for_unclassifiable_test/for_building_code")


    path_to_model = model_path / 'model.pth'


    # set batch size for the dataloader
    batch_size = 32

    # read dictionary of class names and indexes
    with open(model_path / 'class_to_idx.txt') as f:
        data = f.read()

    class_to_idx = ast.literal_eval(data)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())

    # load model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    ) 
     
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval() # enabling the eval mode to test with new samples


    transform = transforms.Compose([
            NewPad(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ])

    # Create dataloader


    _, _, CE_dataloader, _, _, class_names, class_to_idx = create_dataloaders( 
        data_path = data_path,
        transform = transform,
        simple_transform = transform,
        batch_size = batch_size,
        with_unclassifiable=with_unclassifiable
    )
    # read the thresholds

    if not with_unclassifiable:
        thresholds = pd.read_csv(model_path / 'thresholds.csv')['Threshold']
    else:
        thresholds = _

    # call the evaluation function
    eval_df = evaluate_on_test(model = model, dataloader = CE_dataloader, class_names= class_names, with_unclassifiable=with_unclassifiable, thresholds = thresholds)

    print(eval_df)
    
    # write the newly created files
    eval_df.to_csv(base_dir / 'out' / 'test_set_metrics.csv', index=True) #flag

    
    
    
    
    
    
