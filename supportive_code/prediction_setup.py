#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:57:31 2023

@author: forskningskarin
"""
import os
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import ConcatDataset
import shutil
import pandas as pd
import torch
from tqdm import tqdm
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

def find_best_thresholds(model, dataloader, class_names, figures_path):
    thresholds = {}
    f1_scores = {}
    num_classes = len(class_names)
    probabilities = []
    all_labels = []

    thresholds_path = figures_path / 'threshold_graphs' 
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

    for i in tqdm(range(num_classes)):
        print(f"{i} of {num_classes}")
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

            # condition 2: filter preds_i to only where the max_probabilities value is larger than the threshold
            preds_i_thresh = preds_i & (max_probabilities > threshold) # preds_i_thresh is a tensor with True and False values
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

 

def create_predict_dataloader(
    data_path: str, 
    batch_size: int,
    dataset,
    shuffle = False
): 
    #define dataloaders
    batch_size = batch_size
    num_workers = 32
    predict_dataloader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, 
                                 num_workers=num_workers, 
                                 shuffle=shuffle) 
    
    return predict_dataloader


def predict_to_csvs(model, data_loader, dataset, idx_to_class, thresholds_path):
    predictions = []
    threshold_df = pd.read_csv(thresholds_path, index_col=0)

    # 1. Predict on images in dataloader
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs*np.log(1.3), dim=1)
            predicted = torch.argmax(probs, dim=1)
            predicted_prob = torch.max(probs, dim=1).values.cpu().numpy()

            # use thresholds from threshold_df to determine whether to classify each image
            for i in range(len(predicted)):
                class_name = idx_to_class[predicted[i].item()]
                threshold = threshold_df.loc[class_name, 'Threshold']
                if predicted_prob[i].item() < threshold:
                    predicted[i] = -1 # set the class to -1 for images below threshold
            predicted = predicted.cpu().numpy()
            predictions.extend(predicted)

    # Handle both normal ImageFolder and ConcatDataset
    if isinstance(dataset, ConcatDataset):
        all_imgs = []
        for ds in dataset.datasets:
            if hasattr(ds, 'imgs'):
                all_imgs.extend(ds.imgs)
            else:
                raise ValueError("One of the datasets in ConcatDataset has no .imgs attribute")
    else:
        all_imgs = dataset.imgs

    image_names = [os.path.basename(path[0]) for path in all_imgs]
    bin_names = [os.path.basename(os.path.dirname(path[0])) for path in all_imgs]

    df_of_predictions = pd.DataFrame({'bin_name': bin_names, 'image_name': image_names, 'predicted_class': predictions})
    df_of_predictions.replace(-1, "Unclassified", inplace=True)
    df_of_predictions =df_of_predictions.replace({"predicted_class": idx_to_class})
    
    # 4. Group the predictions by bin_name to get the table image_class_table
    counts_per_bin = df_of_predictions.groupby('bin_name')['predicted_class'].value_counts()
    total_counts = df_of_predictions.groupby('bin_name')['predicted_class'].count() # total counts for reach bin
    total_counts_except_unclassified = df_of_predictions.loc[df_of_predictions['predicted_class'] != 'Unclassified'].groupby('bin_name')['predicted_class'].count() # total counts for each bin, excluding unclassified images
    relative_abundance = counts_per_bin / total_counts
    relative_abundance.rename('relative_abundance')
    relative_abundance_without_unclassified = counts_per_bin / total_counts_except_unclassified
    relative_abundance_without_unclassified.rename('relative_abundance_without_unclassifiable')
    image_class_table = pd.DataFrame({'counts_per_bin': counts_per_bin, 'relative_abundance': relative_abundance, 'relative_abundance_without_unclassifiable':relative_abundance_without_unclassified})
    image_class_table = image_class_table.reset_index()
    image_class_table.loc[image_class_table['predicted_class'] == 'Unclassified', 'relative_abundance_without_unclassifiable'] = 0

    return df_of_predictions, image_class_table


import pandas as pd
import torch
from tqdm import tqdm
import os
import time

def predict_to_csvs_streaming(
    model,
    data_loader,
    dataset,
    idx_to_class,
    thresholds_path,
    output_path,
    resume=True,
    save_every_n_batches=10,
    resume_from_paths=None  # 🆕 list of CSVs to resume from
):
    threshold_df = pd.read_csv(thresholds_path, index_col=0)
    predictions = []
    already_predicted_paths = set()

    # 🧠 Load predictions from multiple past CSVs
    if resume_from_paths:
        for path in resume_from_paths:
            if os.path.exists(path):
                print(f"🔁 Loading previously predicted paths from: {path}")
                df_prev = pd.read_csv(path)
                already_predicted_paths.update(df_prev['image_path'].tolist())

    # 🧠 Also include current file if resuming from current folder
    prediction_file = os.path.join(output_path, 'streaming_predictions.csv')
    if resume and os.path.exists(prediction_file):
        df_existing = pd.read_csv(prediction_file)
        already_predicted_paths.update(df_existing['image_path'].tolist())
        predictions.extend(df_existing.to_dict('records'))

    batch_counter = 0
    
    for batch_idx, (images, paths) in enumerate(tqdm(data_loader, desc="Streaming prediction")):
        if not paths:
            print(f"⚠️ Skipping empty batch {batch_idx}")
            continue

        bin_name = os.path.basename(os.path.dirname(paths[0]))
        print(f"📦 Processing: {bin_name} (batch {batch_idx + 1}, {len(paths)} images)")
        
        batch_start = time.time()

        # Inference
        t0 = time.time()
        images = images.to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs * torch.log(torch.tensor(1.3)), dim=1)
            predicted = torch.argmax(probs, dim=1)
            predicted_prob = torch.max(probs, dim=1).values.cpu().numpy()
        t1 = time.time()

        # Post-processing
        for i in range(len(paths)):
            path = paths[i]
            if path in already_predicted_paths:
                continue
            class_idx = predicted[i].item()
            class_name = idx_to_class[class_idx]
            threshold = threshold_df.loc[class_name, 'Threshold']
            if predicted_prob[i] < threshold:
                class_name = "Unclassified"
            bin_name = os.path.basename(os.path.dirname(path))
            image_name = os.path.basename(path)

            pred_entry = {
                'bin_name': bin_name,
                'image_name': image_name,
                'image_path': path,
                'predicted_class': class_name
            }
            predictions.append(pred_entry)

        batch_counter += 1
        if batch_counter % save_every_n_batches == 0:
            pd.DataFrame(predictions).to_csv(prediction_file, index=False)

        t2 = time.time()
        # print(f"🧠 Inference: {t1 - t0:.2f}s | 🛠️ Post-processing: {t2 - t1:.2f}s | Total batch: {time.time() - batch_start:.2f}s")

    # Final save
    pd.DataFrame(predictions).to_csv(prediction_file, index=False)
    print("✅ Final predictions written.")

    return pd.DataFrame(predictions)



def evaluate_on_test(model, dataloader, class_names, thresholds):
    num_classes = len(class_names)
    probabilities = []
    all_labels = []
    f1_scores = []
    recall_scores = []
    precision_scores = []

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

    for i in tqdm(range(num_classes)):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        threshold = thresholds[i]
        # condition 1: filter preds to only contain indices where the predicted class is i
        preds_i = preds == i # preds_i is a tensor with "True" and "False" values

        # condition 2: filter preds_i to only where the max_probabilities value is larger than the threshold
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




def sample_and_sort_images(model, data_loader, dataset, idx_to_class, thresholds_path, output_folder, sample_size=100):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    predictions = []
    threshold_df = pd.read_csv(thresholds_path, index_col=0)
    
    # 1. Predict on images in dataloader
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs * np.log(1.3), dim=1)
            predicted = torch.argmax(probs, dim=1)
            predicted_prob = torch.max(probs, dim=1).values.cpu().numpy()

            # use thresholds from threshold_df to determine whether to classify each image
            for i in range(len(predicted)):
                class_name = idx_to_class[predicted[i].item()]
                threshold = threshold_df.loc[class_name, 'Threshold']
                if predicted_prob[i].item() < threshold:
                    predicted[i] = -1  # set the class to -1 for images below threshold
            predicted = predicted.cpu().numpy()
            predictions.extend(predicted)

    # 2. Make dataframe of predictions per each image id
    image_names = [path[0].split('/')[-1] for path in dataset.imgs]
    bin_names = [path[0].split('/')[-2] for path in dataset.imgs]
    df_of_predictions = pd.DataFrame({
        'bin_name': bin_names, 
        'image_name': image_names, 
        'predicted_class': predictions
    })
    df_of_predictions.replace(-1, "Unclassified", inplace=True)
    df_of_predictions = df_of_predictions.replace({"predicted_class": idx_to_class})

    # 3. Determine the actual sample size
    actual_sample_size = min(sample_size, len(df_of_predictions))

    # 4. Randomly sample images
    sampled_df = df_of_predictions.sample(n=actual_sample_size, replace=False)

    # 5. Sort the images into folders based on their predicted class
    for idx, row in sampled_df.iterrows():
        predicted_class = row['predicted_class']
        image_name = row['image_name']
        bin_name = row['bin_name']
        
        source_path = os.path.join(dataset.root, bin_name, image_name)
        class_folder = os.path.join(output_folder, predicted_class)

        # Ensure the class folder exists
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Copy the image to the class folder
        shutil.copy(source_path, os.path.join(class_folder, image_name))
    
    return sampled_df
