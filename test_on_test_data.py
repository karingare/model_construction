#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:35:56 2023

@author: forskningskarin

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # decides how much info to show from tensorflow
from torchvision import models, transforms, datasets
import torch
from torch import nn
from pathlib import Path
import argparse
from supportive_code.prediction_setup import create_predict_dataloader, evaluate_on_test
import ast
import pandas as pd
from supportive_code.padding import NewPad
from supportive_code.data_setup import create_dataloaders

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")
    figures_path =  base_dir / 'out'

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='test')
    parser.add_argument('--testtype', type=str, help='Specify the type of test data to use (fraction of full set or separate set)', default='fraction')
    parser.add_argument('--data', type=str, help='Specify any specific dataset to use', default='development')
    if parser.parse_args().model == "main":
        model_path = base_dir / 'data' / 'models' /'model_main_240116' 
    elif parser.parse_args().model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209' 
    elif parser.parse_args().model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
    else:
        model_path = base_dir / 'data' / 'models' / parser.parse_args().model 

    if parser.parse_args().data == "development":
        data_path = base_dir / 'data' / 'development'
        unclassifiable_path = base_dir / 'data' / 'development_unclassifiable'
    elif parser.parse_args().data == "syke2022":
        data_path = base_dir / 'data' / 'SYKE_2022' / 'labeled_20201020'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "smhibaltic2023":
        data_path = base_dir / 'data' / 'smhi_training_data_oct_2023' / 'Baltic'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "tangesund":
        data_path = '/proj/common-datasets/SMHI-IFCB-Plankton/version-2/smhi_ifcb_tångesund_annotated_images'
        unclassifiable_path = base_dir / 'data' / 'Unclassifiable from SYKE 2021'

    path_to_model = model_path / 'model.pth'

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
            NewPad(padding_mode=padding_mode),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ])

    # create dataset and dataloader for the data to be predicted on
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    if parser.parse_args().testtype == "fraction":
        _, _, _, test_dataloader, test_with_unclassifiable_dataloader, class_names, class_to_idx = create_dataloaders( 
            data_path = data_path,
            unclassifiable_path = unclassifiable_path,
            transform = transform,
            simple_transform = transform,
            batch_size = batch_size)
    elif parser.parse_args().testtype == "separate":
        test_dataloader = create_predict_dataloader(data_path = data_path, transform = transform, batch_size = batch_size, dataset = dataset)

    # read the thresholds
    thresholds = pd.read_csv(model_path / 'thresholds.csv')['Threshold']

    # call the evaluation function
    eval_df = evaluate_on_test(model, test_dataloader, class_names, thresholds)

    print(eval_df)
    
    # write the newly created files

    eval_df.to_csv(model_path /'figures' /  'test_set_metrics.csv', index=True)

    print(f"[INFO] The test set metrics have been saved to {model_path /'figures' /  'test_set_metrics.csv'}")
    
    
    
    
    
    
