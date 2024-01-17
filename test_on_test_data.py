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
from torchinfo import summary
from pathlib import Path
import argparse
from supportive_code.prediction_setup import create_predict_dataloader, predict_to_csvs, find_best_thresholds, evaluate_on_test
from supportive_code.helper import show_model, create_confusion_matrix
import ast
import pandas as pd
from supportive_code.padding import NewPad
from supportive_code.data_setup import create_dataloaders

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")
    figures_path =  base_dir / 'out'

    data_path = base_dir / 'data' / 'split_datasets' / 'combined_datasets' / 'testWithUnclassifiable'

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--model', type=str, help='Specify model (name of model of main)', default='main')

    if parser.parse_args().model == "main":
        path_to_model = base_dir / 'data' / 'model_main_240116.pth'
    else:
        path_to_model = base_dir / 'data' / parser.parse_args().model

    # set batch size for the dataloader
    batch_size = 32

    # read dictionary of class names and indexes
    with open(base_dir / 'model_construction' / 'supportive_files' / 'class_to_idx.txt') as f:
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

    # create dataset and dataloader for the data to be predicted on
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    data_loader = create_predict_dataloader(data_path = data_path, transform = transform, batch_size = batch_size, dataset = dataset)


    # read the thresholds
    thresholds = pd.read_csv(base_dir / 'out' / 'thresholds.csv')['Threshold']

    # call the evaluation function
    eval_df = evaluate_on_test(model, data_loader, class_names, thresholds)

    print(type(eval_df))
    print(eval_df)
    # write the newly created files
    eval_df.to_csv(base_dir / 'out' / 'test_set_metrics.csv', index=True) #flag

    
    
    
    
    
    
