#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 2 20:59:17 2025
@author: Leela Srinivasan
"""


import os
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


def train_single_default(dataset, model_name, model_class, num_epochs=10, batch_size=32, val_split=0.2):
    """
    Train single CNN to perform 4 class classification under default options.

    Parameters
    ----------
    dataset : TensorDataset
        data.
    model_name : str
        name of model for saving.
    model_class : Class
        ex. CNN.
    num_epochs : int, optional
        The default is 10.
    batch_size : int, optional
        The default is 32.
    val_split : float, optional
        The default is 0.2.

    Returns
    -------
    model : Class
        model trained and ready for evaluation.

    """
    
    # Split training/validation and load data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    print(f"Using device: {device}")


    # Set loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Create directory for saving epochs/metrics
    save_dir = os.path.join("models", model_name+"_training")
    os.makedirs(save_dir, exist_ok=True)
    losses, accuracies = [], []


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        print(f"\nEpoch {epoch+1}/{num_epochs} starting")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total


        # Run validation data
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total


        # Update epoch lists for loss and accuracy
        losses.append({"train": epoch_loss, "val": val_loss})
        accuracies.append({"train": epoch_acc, "val": val_acc})
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")


        # Save models every 5th epoch as per paper
        if (epoch + 1) % 5 == 0:
            model_fname = os.path.join(save_dir, f"model_{model_name}_epoch{epoch + 1}.pth")
            torch.save(model.state_dict(), model_fname)
            print(f"Saved model {model_name} at epoch {epoch + 1} as {model_fname}")


    # Save lossses and accuracies to json file
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"losses": losses, "accuracies": accuracies}, f)
    print(f"Training of model {model_name} complete.")
    return model


def train_single_optimized(dataset, model_name, model_class, num_epochs=10, batch_size=128, val_split=0.2):
    """
    Modify training to include batch_size=128, LR scheduling and smart checkpointing

    Parameters
    ----------
    dataset : TensorDataset
        Preprocessed dataset for training.
    model_name : str
        Name used to save model checkpoints and metrics.
    model_class : class
        The PyTorch model class.
    num_epochs : int
        Number of training epochs. Default 10 from paper
    batch_size : int
        Training batch size.
    save_dir : str
        Directory to save models and metrics.
    val_split : float
        Fraction of dataset to use as validation set.

    Returns
    -------
    model : nn.Module
        The trained model.
    """

    # Split training/validation and load data
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    print(f"Using device: {device}")


    # Loss, optimizer and added LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)


    # Create directory for saving epochs/metrics
    save_dir = os.path.join("models", model_name+"_training")
    os.makedirs(save_dir, exist_ok=True)
    losses, accuracies = [], []
    best_val_loss = float("inf")
    

    # Iterate
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} starting")
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


            # Update running loss
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


        # Update epoch stats
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")


        # Run validation data
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f} Accuracy: {val_acc:.2f}%")


        # Scheduler step
        scheduler.step(val_loss)


        # Track epoch metrics
        losses.append({"train": epoch_loss, "val": val_loss})
        accuracies.append({"train": epoch_acc, "val": val_acc})


        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_fname = os.path.join(save_dir, f"model_{model_name}_best.pth")
            torch.save(model.state_dict(), model_fname)
            print(f"Saved best model at epoch {epoch+1} with val_loss {val_loss:.4f}")


    # Save metrics as json file
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"losses": losses, "accuracies": accuracies}, f)
    print(f"Training of model {model_name} complete.")
    return model


def train_level1(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """
    Train the first block of the dual CNN

    Parameters
    ----------
    model : Class
        ex. CNN.
    train_loader : TensorDataset
        data.
    val_loader : TensorDataset
        data.
    device : str
        device to train on.

    Returns
    -------
    history : dict
        training loss, validation loss and validation accuracy.

    """
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Level1] Epoch {epoch+1}/{epochs}"
              f"  Train Loss: {epoch_loss:.4f}"
              f"  Val Loss: {epoch_val_loss:.4f}"
              f"  Val Acc: {val_acc:.4f}")

    return history


def train_level2(model, train_loader, val_loader, device, epochs=15, lr=1e-3):
    """
    Train the second block of the dual CNN

    Parameters
    ----------
    model : Class
        ex. CNN.
    train_loader : TensorDataset
        x,y.
    val_loader : TensorDataset
        x,y validation.
    device : str
        device to train on.

    Returns
    -------
    history : dict
        training loss, validation loss, and validation accuracy.

    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Level2] Epoch {epoch+1}/{epochs}"
              f"  Train Loss: {epoch_loss:.4f}"
              f"  Val Loss: {epoch_val_loss:.4f}"
              f"  Val Acc: {val_acc:.4f}")

    return history


def plot_metrics(json_path):
    """
    Plot training and validation curves 

    Parameters
    ----------
    json_path : str
        path to saved json file.

    Returns
    -------
    None. Plots.

    """

    # Load saved metrics
    with open(json_path, "r") as f:
        metrics = json.load(f)
    losses = metrics["losses"]
    accuracies = metrics["accuracies"]


    # Extract values into lists
    train_loss = [x["train"] for x in losses]
    val_loss   = [x["val"]   for x in losses]
    train_acc = [x["train"] for x in accuracies]
    val_acc   = [x["val"]   for x in accuracies]


    # Initialize plot
    plt.figure(figsize=(12, 5))
    epochs = range(1, len(train_loss) + 1)


    # Create loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)


    # Create accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()