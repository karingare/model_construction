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
from torchvision import models, transforms
import torch
from torch import nn
from datetime import datetime
from pathlib import Path
import ast
import time
from glob import glob
import argparse

from supportive_code.prediction_setup import predict_to_csvs_streaming
from supportive_code.data_setup import ImageWebDataset
from supportive_code.padding import NewPad

def log_time(start, description):
    elapsed = time.time() - start
    print(f"{description} took {elapsed:.2f} seconds")

# Function to get folders in a specified date range
def get_folders_in_range(base_path, start_date=None, end_date=None):
    filtered = []
    for root, dirs, _ in os.walk(base_path):
        for folder in dirs:
            if folder.startswith('D'):
                try:
                    date_int = int(folder[1:])  # Remove 'D' prefix
                    if start_date and date_int < int(start_date):
                        continue
                    if end_date and date_int > int(end_date):
                        continue
                    full_path = os.path.join(root, folder)
                    filtered.append(full_path)
                except ValueError:
                    continue  # Skip non-date folders
    return filtered

if __name__ == "__main__":  
    print("Starting script")
    start_time = time.time()

    # set up device and base directory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    base_dir = Path("/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin")
    
    # set up argument parser
    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (for example test or all)', default='test')
    parser.add_argument('--model', type=str, help='Specify model (main or a name)', default='test')
    parser.add_argument('--start_date', type=str, help='Start date in YYYYMMDD format', default=None)
    parser.add_argument('--end_date', type=str, help='End date in YYYYMMDD format', default=None)
    args = parser.parse_args()
        




    # Set up data path based on user input
    if args.data == "march2023":
        data_path = base_dir / 'data' / 'ifcb_png_march_2023'
    elif args.data == "tangesund":
        data_path = base_dir / 'data' / 'SMHI_Tangesund_annotated_images'
    elif args.data == "amime":
        data_path = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/AMIME_main_dataset"
    elif args.data == "test":
        data_path = "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/ifcb_data/testing_data_tar/2024"
    else:
        data_path = args.data

    print(f"Data path: {data_path}")

    # Set up model path based on user input, and also some other paths
    if args.model == "main":
        model_path = base_dir / 'data' / 'models' /'main_20240116'
        path_to_model = model_path / 'model.pth'
    elif args.model == "development":
        model_path = base_dir / 'data' / 'models' / 'development_20240209'
        path_to_model = model_path / 'model.pth'
    elif args.model == "syke2022":
        model_path = base_dir / 'data' / 'models' / 'syke2022_20240227'
        path_to_model = model_path / 'model.pth'
    elif args.model == "test":
        model_path = base_dir / 'data' / 'models' / 'amime_test_20250407'
        path_to_model = model_path / 'model.pth'
    else:
        model_path = base_dir / 'data' / 'models' / args.model
        path_to_model = model_path / 'model.pth'

    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")

    # Set up paths
    figures_path= model_path / f"predictions_{now_str}"
    figures_path.mkdir(parents=True, exist_ok=True)

    thresholds_path = model_path / 'thresholds.csv'
    training_info_path = model_path / 'training_info.txt'

    log_time(start_time, "Initial setup")


    # Read info about model training from files
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


    # Load model
    model_start = time.time()
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    ) 
    state_dict = torch.load(path_to_model, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    log_time(model_start, "Loading model")

    # Set up transform
    transform_start = time.time()
    transform = transforms.Compose([
        NewPad(padding_mode=padding_mode),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    log_time(transform_start, "Setting up transform")


    # Create dataloader
    dataloader_start = time.time()
    log_time(dataloader_start, "Creating dataset")

    dataset = ImageWebDataset(data_path, transform=transform)

    def safe_collate(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return torch.empty(0), []
        return torch.utils.data.dataloader.default_collate(batch)

    new_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # ✅ critical for iterable datasets
        collate_fn=safe_collate
    )
    
    log_time(dataloader_start, "Creating dataloader")

    predictions_start = time.time()

    resume_paths = [
        "/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/from_berzelius/ifcb/main_folder_karin/data/models/amime_test_20250407/predictions_20250508_121513/streaming_predictions.csv"
    ]

    df_of_predictions = predict_to_csvs_streaming(
        model=model,
        data_loader=new_data_loader,
        dataset=dataset,
        idx_to_class=idx_to_class,
        thresholds_path=thresholds_path,
        output_path=figures_path,
        resume=True,
        save_every_n_batches=10,
        resume_from_paths=resume_paths
    )

    log_time(predictions_start, "Making predictions")

    log_time(start_time, "Total script execution")

