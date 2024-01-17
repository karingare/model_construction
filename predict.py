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
import argparse

if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (for example test or all)', default='all')
    parser.add_argument('--model', type=str, help='Specify model (main or a name)', default='main')

    if parser.parse_args().data == "march2023":
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'
    elif parser.parse_args().data == "tangesund":
        data_path = base_dir / 'data' / 'SMHI_Tangesund_annotated_images'
    else:
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'

    if parser.parse_args().model == "main":
        path_to_model = base_dir / 'data' / 'model_main_240116.pth'
    else:
        path_to_model = base_dir / 'data' / parser.parse_args().model

    figures_path = base_dir / 'out' 
    
    # set batch size for the dataloader
    batch_size = 6

    # read dictionary of class names and indexes
    with open(base_dir / 'model_construction' / 'supportive_files'/'class_to_idx.txt') as f:
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

    
    # summary(model=model, 
    #         input_size=(1, 3, 180, 180), # make sure this is "input_size", not "input_shape"
    #         # col_names=["input_size"], # uncomment for smaller output
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]
    # )


    transform = transforms.Compose([
            NewPad(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ])

    
    # create dataset and dataloader for the data to be predicted on
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    new_data_loader = create_predict_dataloader(data_path = data_path, transform = transform, batch_size = batch_size, dataset = dataset)
    
    # display some images
    show_model(model= model, dataloader = new_data_loader, class_names = class_names, figures_path = figures_path)
    
    # Make real predictions that will be summarized in csv files
    df_of_predictions, summarized_predictions, summarized_predictions_per_class = predict_to_csvs(model = model, data_loader =new_data_loader, dataset=dataset, idx_to_class=idx_to_class)

    # write the newly created files
    df_of_predictions.to_csv(figures_path / 'individual_image_predictions.csv', index = False) #flag
    summarized_predictions.to_csv(figures_path /'image_class_table.csv') #flag
    summarized_predictions_per_class.to_csv(figures_path / 'tax_class_table.csv') #flag
    

    
    
    
    
    
    
