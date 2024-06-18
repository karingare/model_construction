#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:46:58 2023

@author: forskningskarin
"""

# other things
import warnings
from random import sample 
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
from torchvision import transforms, models
from torch import nn
from torchinfo import summary
import torch.optim as optim
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import argparse
from torch.optim.lr_scheduler import MultiStepLR

# scripts I wrote
from supportive_code.data_setup import create_dataloaders
from supportive_code.engine import train
from supportive_code.helper import plot_loss_curves, show_model, create_confusion_matrix, evaluate, save_model
from supportive_code.padding import NewPad, NewPadAndTransform


if __name__ == "__main__":  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    
    # setting up paths
    base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")

    parser = argparse.ArgumentParser(description='My script description')
    parser.add_argument('--data', type=str, help='Specify data selection (test or all)', default='test')
    parser.add_argument('--num_epochs', type=int, help='Specify data selection (test or all)', default=20)


    if parser.parse_args().data == "test":
        data_path = "/proj/berzelius-2023-48/ifcb/main_folder_karin/data/development"
        unclassifiable_path = "/proj/berzelius-2023-48/ifcb/main_folder_karin/data/development_unclassifiable"
    elif parser.parse_args().data == "syke2022":
        data_path = '/proj/common-datasets/SYKE-plankton_IFCB_2022/20220201/phytoplankton_labeled/labeled_20201020'
        unclassifiable_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "smhibaltic2023":
        data_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/smhi_training_data_oct_2023/Baltic'
        unclassifiable_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Unclassifiable from SYKE 2021'
    elif parser.parse_args().data == "tangesund":
        data_path = '/proj/common-datasets/SMHI-IFCB-Plankton/version-2/smhi_ifcb_tångesund_annotated_images'
        unclassifiable_path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/Unclassifiable from SYKE 2021'

    print(f"[INFO] Using data from {data_path} ")

    # Get the current date and time
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")

    model_save_path = base_dir / 'data' / 'models' / f"model_{now_str}"
    model_save_path.mkdir(parents=True, exist_ok=True)

    figures_path = model_save_path / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    padding = True

    BATCH_SIZE = 32
    NUM_EPOCHS = parser.parse_args().num_epochs

    train_transform = transforms.Compose([
                NewPadAndTransform(),#include more transforms like RandomRotation
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

    # create dataloaders
    train_dataloader, val_dataloader, val_with_unclassifiable_dataloader, test_dataloader, test_with_unclassifiable_dataloader, class_names, class_to_idx = create_dataloaders( 
        data_path = data_path,
        unclassifiable_path = unclassifiable_path,
        transform = train_transform,
        simple_transform = simple_transform,
        batch_size = BATCH_SIZE
    )
    
    num_classes = len(class_names)

    print(f"[INFO] There are {num_classes} classes in the dataset. They include {sample(class_names,1 )}, {sample(class_names,1)} and {sample(class_names,1)}.")
    
    print(f"[INFO] The classes are: {class_names}")
    
    # save class to idx so it can be accessed by other scripts
    f = open(model_save_path / 'class_to_idx.txt' ,"w")
    f.write( str(class_to_idx) )
    f.close()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # create the model
    model_0 = models.resnet18(weights='DEFAULT')
    for param in model_0.parameters():
        param.requires_grad = False


    # Modify the model by adding three linear layers that are trainable
    num_ftrs = model_0.fc.in_features
    model_0.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.Linear(256, 128),
        nn.Linear(128, num_classes)
    )

    # Move the model to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_0.to(device)

    # Print the modified ResNet18 architecture
    # print(model_0)
       
    # Setup loss function (optimizer is set up inside train function)
    loss_fn = nn.CrossEntropyLoss()
    
    # Start the timer
    from timeit import default_timer as timer 
    start_time = timer()
    
    summary(model=model_0, 
            input_size=(1, 3, 180, 180), # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    
    # Train model_0 
    model_0_results = train(model=model_0, 
                            train_dataloader=train_dataloader,
                            test_dataloader=val_dataloader,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS,
                            class_names=class_names
                            )


    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Training complete.")

    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # evaluate with validation_set
    validation_metrics = evaluate(model_0, val_dataloader, train_dataloader, class_names, figures_path)

    #plot loss curves
    plot_loss_curves(model_0_results, figures_path = figures_path)

    # Include it in the model name
    model_name = 'model.pth'
    

    # Save the model
    save_model(model=model_0,
               target_dir=model_save_path,
               model_name=model_name)
    

    # Define information about the model training for info file 
    training_info = {
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'number_of_classes': num_classes,  
        'number_of_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'classes': class_names,
        'training_duration': f'{end_time-start_time:.3f} seconds',
        'data_path': data_path
    }

    # Write the information to a text file
    output_file_path = model_save_path / 'training_info.txt'
    with open(output_file_path, 'w') as f:
        for key, value in training_info.items():
            f.write(f'{key}: {value}\n')

    #summary(model=model_0, 
            #input_size=(1, 3, 180, 180), # make sure this is "input_size", not "input_shape"
            #col_names=["input_size", "output_size", "num_params", "trainable"],
            #col_width=20,
            #row_settings=["var_names"])
    
    show_model(model = model_0, dataloader = val_dataloader, class_names = class_names, figures_path = figures_path)
    
    create_confusion_matrix(model = model_0, test_dataloader = test_dataloader, num_classes = num_classes, class_names = class_names, figures_path = figures_path)