#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:35:56 2023

@author: forskningskarin

depends on reading a model and a class_to_idx text file that is a dictionary between classes and numbers
Gets data from data_path, which should have subfolders for each bin
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from torchvision import models, transforms, datasets
import torch
from torch import nn
from pathlib import Path
from torchinfo import summary
from supportive_code.prediction_setup import create_predict_dataloader, predict_to_csvs, find_best_thresholds
from supportive_code.helper import show_model, create_confusion_matrix, evaluate
import ast
from supportive_code.padding import NewPad
from supportive_code.data_setup import create_dataloaders
from supportive_code.data_setup import CustomImageFolder
import argparse

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (for example test or all)', default='march2023')
    parser.add_argument('--model', type=str, help='Specify model (main or a name)', default='development')

    if parser.parse_args().data == "march2023":
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'
    elif parser.parse_args().data == "tangesund":
        data_path = base_dir / 'data' / 'SMHI_Tangesund_annotated_images'
    elif parser.parse_args().data == "tangesund_unlabeled":
        data_path = '/proj/common-datasets/SMHI-IFCB-Plankton/version-2/smhi_ifcb_tångesund_annotated_images'
    elif parser.parse_args().data == '3m':
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_3_png'
    elif parser.parse_args().data == '6m':
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_6_png'
    elif parser.parse_args().data == '8m':
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_8_png'
    elif parser.parse_args().data == "11m":
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_11_png'
    elif parser.parse_args().data == "13m":
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_13_png'
    elif parser.parse_args().data == "16m":
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/tangesund_png/raw_16'
    else:
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'

    if parser.parse_args().model == "main":
        model_path = base_dir / 'data' / 'models' /'main_20240116'
        path_to_model = model_path / 'model.pth'
    elif parser.parse_args().model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209'
        path_to_model = model_path / 'model.pth'
    elif parser.parse_args().model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
        path_to_model = model_path / 'model.pth'
    else:
        model_path = base_dir / 'data' / 'models' / parser.parse_args().model
        path_to_model = model_path / 'model.pth'
    

    figures_path = model_path / 'predictions' 
    figures_path.mkdir(parents=True, exist_ok=True)

    thresholds_path = model_path / 'thresholds.csv'
    
    # set batch size for the dataloader
    batch_size = 32

    # read dictionary of class names and indexes
    with open(model_path /'class_to_idx.txt') as f:
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
    model.eval() # enabling the eval mode to test with new samples.


    transform = transforms.Compose([
            NewPad(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ])

    
    # create dataset and dataloader for the data to be predicted on
    dataset = CustomImageFolder(root=data_path, transform=transform)
    new_data_loader = create_predict_dataloader(data_path = data_path, transform = transform, batch_size = batch_size, dataset = dataset)
    
    # display some images
    show_model(model= model, dataloader = new_data_loader, class_names = class_names, figures_path = figures_path)
    
    # Make real predictions that will be summarized in csv files
    df_of_predictions, summarized_predictions, summarized_predictions_per_class = predict_to_csvs(model = model, data_loader = new_data_loader, dataset=dataset, idx_to_class=idx_to_class, thresholds_path = thresholds_path)

    # write the newly created files
    df_of_predictions.to_csv(figures_path / 'individual_image_predictions.csv', index = False) #flag
    summarized_predictions.to_csv(figures_path /'image_class_table.csv') #flag
    summarized_predictions_per_class.to_csv(figures_path / 'tax_class_table.csv') #flag
    

    
    
    
    
    
    
