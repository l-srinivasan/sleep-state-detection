#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:45:54 2025
@author: Leela Srinivasan
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


from utils.preprocess import labels_to_numbers, numbers_to_labels


def plot_cm(cm, pred_classes, true_classes, threshold, save_path=None):
    """
    Plot confusion matrix

    Parameters
    ----------
    cm : confusion matrix
        predictions.
    pred_classes : arr
        prediction classes (can include unknown).
    true_classes : arr
        ground truth labels.
    threshold : float
        confidence threshold.
    save_path : str, optional
        path to save image if desired

    Returns
    -------
    None. Plots

    """
    

    # Normalize if safe across classes
    row_sums = cm.sum(axis=1)
    if np.any(row_sums == 0):
        print("Warning: At least one confusion-matrix row has sum = 0. Normalization skipped.")
        cm_normalized = cm.astype(float)
    else:
        cm_normalized = cm.astype(float) / row_sums[:, np.newaxis]


    # Plot CM
    plt.figure(figsize=(7, 4))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=numbers_to_labels([l if l != -1 else 4 for l in pred_classes]),
        yticklabels=numbers_to_labels(true_classes)
    )

    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (CL={threshold})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


    
def evaluate_single_cnn(model, model_name, test_dataset, device="cpu", threshold=0.9):
    """
    Principle evaluation function

    Parameters
    ----------
    model : class
        ex. CNN.
    model_name : str
        name of model for saving.
    test_dataset : TensorDataset
        test data in TensorDataset format

    Returns
    -------
    cm : confusion matrix
        predictions for each class.
    unknown_df : df
        list of unknowns/unclassified samples.
    metrics_df : df
        df containing training and validation curve information.

    """

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    all_preds, all_labels, unknown_indices, low_conf_preds, unknown_true_labels = [], [], [], [], []


    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Make prediction and pull true value
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            true = labels.item()


            # Check confidence against threshold
            if probs.max().item() < threshold:
                unknown_indices.append(idx)
                unknown_true_labels.append(true)
                low_conf_preds.append(pred)
                pred = -1 # Set to unknown

            all_preds.append(pred)
            all_labels.append(true)
    
    
    # Create directory for saving epochs/metrics
    save_dir = os.path.join("models", model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    
    # Save unknown sample info to df for autoencoder
    unknown_df = pd.DataFrame({
        "index": unknown_indices,
        "y_true": unknown_true_labels,
        "low_conf_pred": low_conf_preds
    })
    f_unknown = os.path.join(save_dir, f"unknown_df_{model_name}.pkl")
    unknown_df.to_pickle(f_unknown)
    print(f"[{model_name}] Unknown count: {len(unknown_indices)}")
    print(f"Saved unknown classifcations to {f_unknown}")
    
    
    # Save all predictions
    all_preds_df = pd.DataFrame({
        "index": list(range(len(all_preds))),
        "y_true": all_labels,
        "y_pred": all_preds
    })
    f_allpreds = os.path.join(save_dir, f"all_preds_df_{model_name}.pkl")
    all_preds_df.to_pickle(f_allpreds)
    print(f"Saved all predictions to {f_allpreds}")
    

    # Compute confusion matrix
    true_classes = sorted(set(all_labels))
    pred_classes = true_classes + [-1]
    cm_full = confusion_matrix(all_labels, all_preds, labels=pred_classes)
    cm = cm_full[:-1, :] # Drop unknown row but keep column
    f_cm = os.path.join(save_dir, "confusion_matrix.png")
    plot_cm(cm, pred_classes, true_classes, threshold, save_path=f_cm)


    # Compute metrics
    mask = np.array(all_preds) != -1
    filtered_y_true = np.array(all_labels)[mask]
    filtered_y_pred = np.array(all_preds)[mask]

    precision = precision_score(filtered_y_true, filtered_y_pred, average=None, labels=true_classes)
    recall = recall_score(filtered_y_true, filtered_y_pred, average=None, labels=true_classes)
    f1 = f1_score(filtered_y_true, filtered_y_pred, average=None, labels=true_classes)
    class_names = numbers_to_labels(true_classes)

    metrics_df = pd.DataFrame({
        "class": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })
    
    f_metrics = os.path.join(save_dir, f"metrics_df_{model_name}.pkl")
    metrics_df.to_pickle(f_metrics)
    print(f"Saved metrics to {f_metrics}")
    
    return cm, unknown_df, metrics_df


def evaluate_hierarchical(
    model_primary, model_secondary, dataloader,
    device="cpu", threshold=0.9):
    """

    Parameters
    ----------
    model_primary : model
        First CNN.
    model_secondary : model
        Second CNN.
    dataloader : dataloader
        test data to evaluate.

    Returns
    -------
    cm_primary : confusion matrix
        predictions from the first CNN.
    cm_secondary : confusion matrix
        predictions from the second CNN.

    """
    
    # Evaluate models and initialize lists
    model_primary.eval()
    model_secondary.eval()
    all_primary_preds, all_primary_labels, all_secondary_preds, all_secondary_labels = [], [], [], []


    with torch.no_grad():
        for x, primary_label, secondary_label in dataloader:
            x = x.to(device)


            # Primary prediction
            logits_primary = model_primary(x)
            probs_primary = F.softmax(logits_primary, dim=1)
            pred_primary = torch.argmax(probs_primary, dim=1)
            all_primary_preds.extend(pred_primary.cpu().tolist())
            all_primary_labels.extend(primary_label.cpu().tolist())


            # Secondary prediction
            for i in range(len(x)):
                if pred_primary[i].item() == 2:  # microarousal parent
                    logits_secondary = model_secondary(x[i].unsqueeze(0))
                    probs_secondary = F.softmax(logits_secondary, dim=1)
                    pred_secondary = torch.argmax(probs_secondary, dim=1).item()
                    all_secondary_preds.append(pred_secondary)
                    all_secondary_labels.append(secondary_label[i].item())


    # Compute confusion matrices
    cm_primary = confusion_matrix(all_primary_labels, all_primary_preds)
    cm_secondary = confusion_matrix(all_secondary_labels, all_secondary_preds)
    return cm_primary, cm_secondary



def evaluate_and_plot(y_true, y_pred, class_names):
    """
    Evaluate classification report and print confusion matrix

    Parameters
    ----------
    y_true : arr
        ground truth labels  .
    y_pred : arr
        predictions.
    class_names : list
        list of class names (not encodings).

    Returns
    -------
    None.

    """
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_metrics_across_thresholds(threshold_list, classes, precision_array, recall_array, f1_array):
    """
    Sister function to summarize_metrics below. 

    Parameters
    ----------
    threshold_list : list
        list of thresholds. (ex. [0.5, 0.6])
    classes : array
        array of classes.
    precision_array : array
        generated by summarize_metrics
    recall_array : array
        generated by summarize_metrics
    f1_array : array
        generated by summarize_metrics

    Returns
    -------
    None. Plots.

    """
    metrics_arrays = {"Precision": precision_array, "Recall": recall_array, "F1 Score": f1_array}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric_name, arr) in zip(axes, metrics_arrays.items()):
        for i, cls in enumerate(classes):
            ax.plot(threshold_list, arr[i], marker='o', label=cls)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs Threshold")
        ax.set_xticks(threshold_list)
        ax.set_ylim(0.6, 1.05)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
    
    
def summarize_metrics(model_name, threshold_list):
    """
    Summarize precision, recall and f-1 score for a given model across tested thresholds.

    Parameters
    ----------
    model_name : str
        model name as per naming convention (ex. cnn_standard_optimized).
    threshold_list : list
        list of thresholds. (ex. [0.5, 0.6])

    Returns
    -------
    None. Plots metrics

    """
    
    classes = ["Awake", "Slow Up-Down", "Asynch MA", "Slow MA"]
    metrics = ["precision", "recall", "f1_score"]
    
    # Initialize dicts to store results
    results = {metric: {cls: [] for cls in classes} for metric in metrics}
    
    for threshold in threshold_list:
      f_folder = f"models/{model_name}_{threshold}/"
      f_metrics = f_folder + f"metrics_df_{model_name}_{threshold}.pkl"
      metrics_df = pd.read_pickle(f_metrics)
    
      for cls in classes:
            row = metrics_df[metrics_df["class"] == cls].iloc[0]
            for metric in metrics:
                results[metric][cls].append(row[metric])
    
    precision_array = np.array([results["precision"][cls] for cls in classes])
    recall_array = np.array([results["recall"][cls] for cls in classes])
    f1_array = np.array([results["f1_score"][cls] for cls in classes])

    # Call the plotting function
    plot_metrics_across_thresholds(threshold_list, classes, precision_array, recall_array, f1_array)
    

def get_unknown_percentages(model_list, test_indices, metadata):
    """
    Get percentages of unknown for each class across models.

    Parameters
    ----------
    model_list : list
        List of model names.
    test_indices : array
        indices from the full dataset that belong to the test set.
    metadata : df
        full df of metadata.

    Returns
    -------
    result_df : df
        percentage information.

    """
    # Compute full-test counts once (same for all models)
    arr = labels_to_numbers(metadata.iloc[test_indices].state)
    full_test_counts = pd.Series(arr).value_counts().sort_index()
    result_df = pd.DataFrame(index=full_test_counts.index)

    # Iterate and read unknown test indices
    for model in model_list:
        f_unknown = f"models/{model}/unknown_df_{model}.pkl"
        unknown_df = pd.read_pickle(f_unknown)

        # Count unknowns for this model and build percentages of whole test set
        unknown_counts = unknown_df["y_true"].value_counts().sort_index()
        percentages = (unknown_counts.reindex(full_test_counts.index, fill_value=0) /
                       full_test_counts) * 100

        # Append to output
        result_df[model] = percentages
    return result_df


def plot_train_val_loss():
    """
    Plot Training and Validation Loss and Accuracy curves for the final 3 models

    Returns
    -------
    None. Plots curves.

    """
    # Using all final models
    model_names = ["Model 1", "Model 2", "Model 3"]
    model_codes = ["cnn_standard_default",
                   "cnn_standard_optimized",
                   "cnn_modified_optimized"]
    f_list = [f"models/{mod}_training/{mod}_metrics.json" for mod in model_codes]

    # Iterate through models
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), sharex=True)
    for i, (name, fpath) in enumerate(zip(model_names, f_list)):
        x = pd.read_json(fpath)
        df = x.copy()
    
        # Separate stats from json saved dict
        df["train_loss"] = df["losses"].apply(lambda d: d["train"])
        df["val_loss"]   = df["losses"].apply(lambda d: d["val"])
        df["train_acc"]  = df["accuracies"].apply(lambda d: d["train"])
        df["val_acc"]    = df["accuracies"].apply(lambda d: d["val"])
        df["epoch"] = df.index + 1
    
        # Loss plot
        ax_loss = axes[i, 0]
        ax_loss.plot(df["epoch"], df["train_loss"], label="Train")
        ax_loss.plot(df["epoch"], df["val_loss"], label="Validation")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{name}: Loss")
        ax_loss.legend()
    
        # Acc plot
        ax_acc = axes[i, 1]
        ax_acc.plot(df["epoch"], df["train_acc"], label="Train")
        ax_acc.plot(df["epoch"], df["val_acc"], label="Validation")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title(f"{name}: Accuracy")
        ax_acc.legend()
    
    # Shared x-axis label
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    
    # Overall fig title
    fig.suptitle("Training and Validation Performance Across Models", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
