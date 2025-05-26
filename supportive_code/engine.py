#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Contains functions for training and testing a PyTorch model. The the main function is train, and it it supported by four functions: train_step, test_step, freeze_layers and unfreeze_layers.
"""
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import torch
from tqdm.auto import tqdm
from torch import nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from typing import List


device = "cuda" if torch.cuda.is_available() else "cpu"

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               lastEpoch: bool,
               class_names):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy, train_precision, train_recall, train_f1_score).
        For example:
        
        (0.1112, 0.8743, 0.8392, 0.9081, 0.8724)
    """
    
    # Put model in train mode
    model.train()
    # Initialize variables to store metrics
    train_loss, train_acc, train_precision, train_recall, train_f1_score = 0, 0, 0, 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Calculate precision, recall, and F1 score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y.cpu().numpy(), y_pred_class.cpu().numpy(), average='weighted')
        train_precision += precision
        train_recall += recall
        train_f1_score += f1_score



    # Adjust metrics to get average loss and metrics per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_precision = train_precision / len(dataloader)
    train_recall = train_recall / len(dataloader)
    train_f1_score = train_f1_score / len(dataloader)


    return train_loss, train_acc, train_precision, train_recall, train_f1_score


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              class_names,
              lastEpoch: bool):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss, test accuracy, precision, recall, and F1-score values
    test_loss, test_acc, precision, recall, f1_score = 0, 0, 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            # Calculate and accumulate precision, recall, and f1-score
            precision_batch, recall_batch, f1_score_batch, _ = precision_recall_fscore_support(y.cpu().numpy(), test_pred_labels.cpu().numpy(), average='weighted')
            precision += precision_batch
            recall += recall_batch
            f1_score += f1_score_batch

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    precision = precision / len(dataloader)
    recall = recall / len(dataloader)
    f1_score = f1_score / len(dataloader)

    return test_loss, test_acc, precision, recall, f1_score


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          class_names, 
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy and f1 score metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              ...

    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              ...

    """
    # 2. Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "train_prec": [], "train_rec": [], "train_f1": [],
               "test_loss": [], "test_acc": [], "test_prec": [], "test_rec": [], "test_f1": []}
    # 3. Plan for changing learning rates and what layers are being trained
    learning_rates = {
        "fc": 0.01,
        "layer4": 0, "rest":0
    }

    # Define the optimizer for each layer with its corresponding learning rate
    optimizer_params = [
        {"params": model.fc.parameters(), "lr": learning_rates["fc"]},
        {"params": model.layer4.parameters(), "lr": learning_rates["layer4"]},
        {"params": model.layer3.parameters(), "lr": learning_rates["rest"]},
        {"params": model.layer2.parameters(), "lr": learning_rates["rest"]},
        {"params": model.layer1.parameters(), "lr": learning_rates["rest"]},
    ]
    optimizer = optim.Adam(optimizer_params)
    lastEpoch = False
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_step(model=model,
                                                     dataloader=train_dataloader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer, 
                                                     lastEpoch=lastEpoch,
                                                     class_names=class_names)
        test_loss, test_acc, test_prec, test_rec, test_f1 = test_step(model=model,   
                                                     dataloader=test_dataloader, 
                                                     loss_fn=loss_fn, 
                                                     device = device,
                                                     class_names=class_names, 
                                                     lastEpoch=lastEpoch)
        
        # 4. Print out what's happening
        print(
            f"\nEpoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_prec: {train_prec:.4f} | "
            f"train_rec: {train_rec:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"\n           test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_prec: {test_prec:.4f} | "
            f"test_rec: {test_rec:.4f} | "
            f"test_f1: {test_f1:.4f}")
        print(f"learning rates: {learning_rates}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_prec"].append(train_prec)
        results["train_rec"].append(train_rec)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_prec"].append(test_prec)
        results["test_rec"].append(test_rec)
        results["test_f1"].append(test_f1)

        # 6. Update learning rate if neccessary
        if epoch == 4:
            unfreeze_layers(model, ['layer4'])
            learning_rates = {
                "fc": 0.005,
                "layer4": 0.001, "rest":0
            }

            optimizer_params = [
                {"params": model.fc.parameters(), "lr": learning_rates["fc"]},
                {"params": model.layer4.parameters(), "lr": learning_rates["layer4"]},
                {"params": model.layer3.parameters(), "lr": learning_rates["rest"]},
                {"params": model.layer2.parameters(), "lr": learning_rates["rest"]},
                {"params": model.layer1.parameters(), "lr": learning_rates["rest"]},
            ]
            optimizer = optim.Adam(optimizer_params)

        if epoch == 14:
            unfreeze_layers(model, ['layer3','layer2','layer1'])
            learning_rates = {
                "fc": 0.00025,
                "layer4": 0.001, "rest":0.0001
            }

            optimizer_params = [
                {"params": model.fc.parameters(), "lr": learning_rates["fc"]},
                {"params": model.layer4.parameters(), "lr": learning_rates["layer4"]},
                {"params": model.layer3.parameters(), "lr": learning_rates["rest"]},
                {"params": model.layer2.parameters(), "lr": learning_rates["rest"]},
                {"params": model.layer1.parameters(), "lr": learning_rates["rest"]},
            ]
            optimizer = optim.Adam(optimizer_params)
        if epoch == epochs-2:
            lastEpoch = True

    return results

def freeze_layers(model: torch.nn.Module, layer_names: List[str]):
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False

def unfreeze_layers(model: torch.nn.Module, layer_names: List[str]):
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True





