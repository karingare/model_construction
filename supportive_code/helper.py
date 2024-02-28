#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:51:44 2023

@author: forskningskarin
"""

from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def plot_loss_curves(results: Dict[str, List[float]], figures_path):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "train_f1": [...],
             "train_prec": [...],
             "train_rec": [...],
             "test_loss": [...],
             "test_acc": [...],
             "test_f1": [...],
             "test_prec": [...],
             "test_rec": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Get the f1 values of the results dictionary (training and test)
    f1 = results['train_f1']
    test_f1 = results['test_f1']
    
    # Get the precision values of the results dictionary (training and test)
    precision = results['train_prec']
    test_precision = results['test_prec']
    
    # Get the recall values of the results dictionary (training and test)
    recall = results['train_rec']
    test_recall = results['test_rec']

    # Figure out how many epochs there were
    epochs = range(len(loss))

    # Setup a plot
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
    
    # Plot loss
    #axs[0, 0].plot(epochs, loss, label='train_loss')
    #axs[0, 0].plot(epochs, test_loss, label='test_loss')
    #axs[0, 0].set_title('Loss')
    #axs[0, 0].set_xlabel('Epochs')
    #axs[0, 0].legend()
    #plt.ylim([0, 1])

    # Plot accuracy
    axs[0, 0].plot(epochs, accuracy, label='Training set', color='hotpink')
    axs[0, 0].plot(epochs, test_accuracy, label='Validation set', color='purple')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].legend()
    axs[0, 0].axvline(x=5, linestyle='dotted', color='deepskyblue')
    axs[0, 0].axvline(x=15, linestyle='dotted', color='deepskyblue')

    # Plot f1 score
    axs[1, 0].plot(epochs, f1, label='Training set', color='hotpink')
    axs[1, 0].plot(epochs, test_f1, label='Validation set', color='purple')
    axs[1, 0].set_title('F1 Score')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].legend()
    axs[1, 0].axvline(x=5, linestyle='dotted', color='deepskyblue')
    axs[1, 0].axvline(x=15, linestyle='dotted', color='deepskyblue')

    # Plot precision
    axs[1, 1].plot(epochs, precision, label='Training set', color='hotpink')
    axs[1, 1].plot(epochs, test_precision, label='Validation set', color='purple')
    axs[1, 1].set_title('Precision')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].legend()
    axs[1, 1].axvline(x=5, linestyle='dotted', color='deepskyblue')
    axs[1, 1].axvline(x=15, linestyle='dotted', color='deepskyblue')

    # Plot recall
    axs[0, 1].plot(epochs, recall, label='Training set', color='hotpink')
    axs[0, 1].plot(epochs, test_recall, label='Validation set', color='purple')
    axs[0, 1].set_title('Recall')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].legend()
    axs[0, 1].axvline(x=5, linestyle='dotted', color='deepskyblue')
    axs[0, 1].axvline(x=15, linestyle='dotted', color='deepskyblue')
    

    # add some space between subplots
    plt.subplots_adjust(hspace=0.5)

    # Save the figure
    full_path = figures_path / 'loss_curves.png'
    plt.savefig(full_path)

    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def show_model(model, dataloader, class_names, figures_path, num_images=6):
    plt.figure()
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(6):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(tf.transpose(inputs.cpu().data[j], perm=([1, 2, 0])))

            if images_so_far == num_images:
                    model.train(mode=was_training)
                    break
            model.train(mode=was_training)

	# 4. Save the figure
    full_path = figures_path / 'show_model.png'
    plt.savefig(full_path)


def create_confusion_matrix(model, test_dataloader, num_classes, class_names, figures_path):
    y_preds = []
    y_true = []
    model.eval()
    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            # Send data and targets to target device
            inputs, labels = inputs.to(device), labels.to(device)
            # Do the forward pass
            outputs = model(inputs)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            _, y_pred = torch.max(outputs, 1)
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
            y_true.append(labels.cpu())

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    y_true_tensor = torch.cat(y_true)
    
    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                         target=y_true_tensor)
    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=class_names, # turn the row and column labels into class names
        figsize=(40, 28)
    );
    
	# 4. Save the figure
    full_path = figures_path / 'confusion_matrix.png'
    plt.savefig(full_path)


def evaluate(model, dataloader, train_dataloader, class_names, figures_path):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_predictions, labels=range(len(class_names)))


    # Iterate over the DataLoader
    # Initialize a dictionary to store the class counts
    class_counts = {class_name: 0 for class_name in train_dataloader.dataset.dataset.classes}

    # Iterate over the DataLoader
    for inputs, labels in train_dataloader:
        # Iterate over each label in the batch
        for label in labels:
            # Increment the count for this class
            class_name = train_dataloader.dataset.dataset.classes[label.item()]
            class_counts[class_name] += 1

    df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Test count": support
    })

    df['Train Count'] = df['Class'].map(class_counts)

    df.set_index("Class", inplace=True)
    df.index.name = None
    df.columns.name = None
    df.to_csv(figures_path / "validation_metrics.csv")
    return df
