#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:57:31 2023

@author: forskningskarin
"""
from torchvision import transforms, datasets
import os
from torch.utils.data import DataLoader
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, f1_score
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_best_svm_thresholds(model, dataloader, class_names, figures_path):
    thresholds = {}
    f1_scores = {}
    num_classes = len(class_names)
    probabilities = []
    all_labels = []

    thresholds_path = figures_path / 'threshold_graphs_svm' 
    thresholds_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs*np.log(1.3), dim=1)
            probabilities.append(probs)
            all_labels += labels
    probabilities = torch.cat(probabilities)
    max_probabilities, preds = torch.max(probabilities.cpu(), dim=1) # pred is a tensor of numbers (assigned labels)
    tensor_labels = torch.stack(all_labels)

    # gather the image ids
    paths_and_labs = np.array(dataloader.dataset.dataset.samples)
    selected_section_for_dataloader = paths_and_labs[dataloader.dataset.indices]

    image_paths = selected_section_for_dataloader[:, 0]
    image_ids = [os.path.basename(path) for path in image_paths]  

    print('small section: of the image ids for the images in the val set', image_ids[:5])
    # print('image ids shape', len(image_ids)) # 8324 images in the val set


    ############ MEMEBER SHIP SCORES ################
    # read the class membership scores
    class_membership_scores = pd.read_csv('/proj/berzelius-2023-48/dd2424/project_v_2/data/dataframes/tangesund_one_class_svm_scores.csv')

    # normalize the class membership scores so they are between 0 and 1 except for the 'Image File Name' column

    for cls in class_names:
        if cls in class_membership_scores.columns:
            class_membership_scores[cls] = (class_membership_scores[cls] - class_membership_scores[cls].min()) / (class_membership_scores[cls].max() - class_membership_scores[cls].min())

    image_ids = pd.Series(image_ids)

    # Filter the DataFrame to include only rows with 'Image File Name' in image_ids
    filtered_df = class_membership_scores[class_membership_scores['Image File Name'].isin(image_ids)]

    print('number of images in the val dataset where there are features available:',image_ids.isin(class_membership_scores['Image File Name']).sum())

    # Reorder the filtered DataFrame based on the order in image_ids
    # Use a categorical data type to preserve the order
    filtered_df.loc[:, 'Image File Name'] = pd.Categorical(filtered_df['Image File Name'], categories=image_ids, ordered=True)

    # Sort the DataFrame by the 'Image File Name' column
    reordered_df = filtered_df.sort_values('Image File Name').reset_index(drop=True)

    # Display the first few rows of the reordered DataFrame
    print('after reordering:')

    # Display the first few rows of the reordered DataFrame
    print(reordered_df.head())
    print('shape: ', reordered_df.shape)

    scores_df = reordered_df


    # this reordered df includes the class memberships scores for the images in the val set (or at least some)
    # order the class membership scores according to the image ids

    ############# PREDICTION SCORES ################
    # preds is the predicted class for each image in the val set
    # subset preds to only contain the indices of the images in the val set that have features

    # preds is a tensor of numbers (assigned labels)
    preds = preds[image_ids.isin(scores_df['Image File Name'])]
    tensor_labels = tensor_labels[image_ids.isin(scores_df['Image File Name'])]

    # also subset the 
    for i in range(num_classes):
        cls = class_names[i]
        print(f"{i} of {num_classes}")
        print(cls)
        if cls not in class_membership_scores.columns:
            print(f"Class {cls} not found in class membership scores")
            continue
        

        current_class_scores = scores_df[cls]
        print('current scores shape:', current_class_scores.shape)
        print('current scores:', current_class_scores.head())
        current_class_scores_tensor = torch.tensor(current_class_scores.values)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        f1_max = -1
        threshold_best = 0
        precision_for_plot = []
        recall_for_plot = []
        f1_for_plot = []

        for threshold in np.arange(1.0, 0.0, -0.001):
            # condition 1: filter preds to only contain indices where the predicted class is i
            preds_i = preds == i # preds_i is a tensor with "True" and "False" values

            # condition 2: filter preds_i to only where the class membership score value is larger than the threshold
            preds_i_thresh = preds_i & (current_class_scores_tensor > threshold) # preds_i_thresh is a tensor with True and False values


            true_positives = torch.sum(preds_i_thresh.cpu() & (tensor_labels.cpu() == i)).item()
            false_positives = torch.sum(preds_i_thresh.cpu() & (tensor_labels.cpu() != i)).item()
            false_negatives = torch.sum((preds_i_thresh == False) & (tensor_labels.cpu() == i)).item()

            precision = true_positives / (true_positives + false_positives + 1e-15)
            recall = true_positives / (true_positives + false_negatives + 1e-15)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

            precision_for_plot.append(precision)
            recall_for_plot.append(recall)
            f1_for_plot.append(f1)
            
            if f1 > f1_max:	
                f1_max = f1
                threshold_best = threshold
        thresholds[class_names[i]] = threshold_best
        f1_scores[class_names[i]] = f1_max

        #plot the f1, prec, recall
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(np.arange(1.0, 0.0, -0.001), f1_for_plot)
        ax.plot(np.arange(1.0, 0.0, -0.001), precision_for_plot)
        ax.plot(np.arange(1.0, 0.0, -0.001), recall_for_plot)
        plt.xlabel('Threshold')
        plt.ylabel('Metric')
        fig_path = thresholds_path / f'{class_names[i]}.png'

        plt.legend([f'F1 Score for {class_names[i]}', f'Precision for {class_names[i]}', f'Recall for {class_names[i]}'])
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

    df_thresholds = pd.DataFrame.from_dict(thresholds, orient='index', columns=['Threshold'])
    df_f1_scores = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1 Score'])
    df = pd.concat([df_thresholds, df_f1_scores], axis=1)
    df.index = class_names

    return df



def evaluate_on_test_svm(model, dataloader, class_names, class_membership_thresholds):
    num_classes = len(class_names)
    probabilities = []
    all_labels = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    all_file_names = []


    with torch.no_grad():
        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs*np.log(1.3), dim=1)
            probabilities.append(probs)
            all_labels += labels

    paths_and_labs = np.array(dataloader.dataset.dataset.samples)
    selected_section_for_dataloader = paths_and_labs[dataloader.dataset.indices]

    image_paths = selected_section_for_dataloader[:, 0]
    image_ids = [os.path.basename(path) for path in image_paths]
    print('small section: of the image ids', image_ids[:5])

    probabilities = torch.cat(probabilities)
    max_probabilities, preds = torch.max(probabilities.cpu(), dim=1) # pred is a tensor of numbers (assigned labels)
    tensor_labels = torch.stack(all_labels)


    # make dataframe organized in the same way, based on image ids and class membership values from another dataframe, integrated with this one
    for i in tqdm(range(num_classes)):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        threshold = class_membership_thresholds[i]

        
        # condition 1: filter preds to only contain indices where the predicted class is i
        preds_i = preds == i # preds_i is a tensor with "True" and "False" values

        # condition 2: filter preds_i to only cases where the class_membership value is larger than the threshold
        # find class membership value

        preds_i_thresh = preds_i & (max_probabilities > threshold) # preds_i_thresh is a tensor with True and False values
        
        true_positives = torch.sum(preds_i_thresh.cpu() & (tensor_labels.cpu() == i)).item()
        false_positives = torch.sum(preds_i_thresh.cpu() & (tensor_labels.cpu() != i)).item()
        false_negatives = torch.sum((preds_i_thresh == False) & (tensor_labels.cpu() == i)).item()

        precision = true_positives / (true_positives + false_positives + 1e-15)
        recall = true_positives / (true_positives + false_negatives + 1e-15)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    df = pd.DataFrame({'Precision': precision_scores, 'Recall':recall_scores, 'F1': f1_scores})
    df.index = class_names
    return df






