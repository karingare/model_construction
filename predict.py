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
from datetime import datetime
from pathlib import Path
from torchinfo import summary
from supportive_code.prediction_setup import create_predict_dataloader, predict_to_csvs, find_best_thresholds
from supportive_code.helper import show_model, create_confusion_matrix, evaluate
import ast
from supportive_code.padding import NewPad
from supportive_code.data_setup import create_dataloaders, CustomImageFolderAllImages
import argparse
import time

def log_time(start, description):
    elapsed = time.time() - start
    print(f"{description} took {elapsed:.2f} seconds")

if __name__ == "__main__":  
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path("/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (for example test or all)', default='march2023')
    parser.add_argument('--model', type=str, help='Specify model (main or a name)', default='development')
    
    args = parser.parse_args()

    if args.data == "march2023":
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'
    elif args.data == "tangesund":
        data_path = base_dir / 'data' / 'SMHI_Tangesund_annotated_images'
    else:
        data_path = args.data

    if args.model == "main":
        model_path = base_dir / 'data' / 'models' /'main_20240116'
        path_to_model = model_path / 'model.pth'
    elif args.model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209'
        path_to_model = model_path / 'model.pth'
    elif args.model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
        path_to_model = model_path / 'model.pth'
    else:
        model_path = base_dir / 'data' / 'models' / args.model
        path_to_model = model_path / 'model.pth'

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    figures_path= model_path / f"predictions_{now_str}"
    figures_path.mkdir(parents=True, exist_ok=True)

    thresholds_path = model_path / 'thresholds.csv'
    training_info_path = model_path / 'training_info.txt'

    log_time(start_time, "Initial setup")

    training_info_start = time.time()
    training_info = {}
    with open(training_info_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            try:
                value = eval(value)
            except (SyntaxError, NameError):
                pass
            training_info[key] = value

    padding_mode = training_info.get('padding_mode')
    log_time(training_info_start, "Reading training info")

    batch_size = 32

    class_to_idx_start = time.time()
    with open(model_path /'class_to_idx.txt') as f:
        data = f.read()
    class_to_idx = ast.literal_eval(data)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    class_names = list(class_to_idx.keys())
    log_time(class_to_idx_start, "Reading class to idx")

    model_start = time.time()
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    ) 
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    model.eval()
    log_time(model_start, "Loading model")

    transform_start = time.time()
    transform = transforms.Compose([
        NewPad(padding_mode=padding_mode),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    log_time(transform_start, "Setting up transform")

    dataloader_start = time.time()
    dataset = CustomImageFolderAllImages(root=data_path, transform=transform)
    log_time(dataloader_start, "Creating dataset")
    new_data_loader = create_predict_dataloader(data_path=data_path, batch_size=batch_size, dataset=dataset)
    log_time(dataloader_start, "Creating dataloader")

    show_model_start = time.time()
    show_model(model=model, dataloader=new_data_loader, class_names=class_names, figures_path=figures_path)
    log_time(show_model_start, "Showing model")

    predictions_start = time.time()
    df_of_predictions, summarized_predictions = predict_to_csvs(model=model, data_loader=new_data_loader, dataset=dataset, idx_to_class=idx_to_class, thresholds_path=thresholds_path)
    log_time(predictions_start, "Making predictions")

    csv_write_start = time.time()
    df_of_predictions.to_csv(figures_path / 'individual_image_predictions.csv', index=False)
    summarized_predictions.to_csv(figures_path /'image_class_table.csv')
    log_time(csv_write_start, "Writing CSV files")

    log_time(start_time, "Total script execution")

    
    
    
    
    
