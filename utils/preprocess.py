#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 2 18:13:13 2025
@author: Leela Srinivasan
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
from torch.utils.data import random_split, TensorDataset


def load_data():
    """
    Read data files from data directory. Return data and metadata dfs.

    Returns
    -------
    data : df
        Time series LFP data.
    metadata : df
        Corresponding metadata.

    """
    data_dir = os.path.join(os.getcwd(), "data")
    metadata_fp = os.path.join(data_dir, "metadata_states.pkl")
    data_fp = os.path.join(data_dir, "data_states.npy")
    
    data = np.load(data_fp)
    with open(metadata_fp, "rb") as f:
      metadata = pd.read_pickle(f)
    return data, metadata


def reload_test_data(data, metadata):
    test_indices = torch.load("models/test_dataset_indices.pth")
    test_data = data[test_indices]
    test_metadata = metadata.iloc[test_indices]
    return test_data, test_metadata


def normalize_data(data):
    """

    Parameters
    ----------
    data : df
        Time series data.

    Returns
    -------
    data : df
        Normalize time series data.

    """
    return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)


def labels_to_numbers(labels):
    """
    Encode brain states to classes

    Parameters
    ----------
    labels : array
        state column of metadata file.

    Returns
    -------
    encoded_labels: array
        encoded states.

    """
    mapping = {
        'awake': 0,
        'slow_updown': 1,
        'asynch_MA': 2,
        'slow_MA': 3, 
        'unknown': 4, 
    }
    
    encoded_labels = [mapping[label] for label in labels]
    return encoded_labels


def numbers_to_labels(encoded_labels):
    """
    Decode numeric state labels back to their string names.

    Parameters
    ----------
    encoded_labels : array-like
        Numeric encoded labels

    Returns
    -------
    labels : list of str
        Decoded state names. 
    """

    inverse_mapping = {
        0: 'Awake',
        1: 'Slow Up-Down',
        2: 'Asynch MA',
        3: 'Slow MA',
        4: 'Unknown',
    }

    labels = [inverse_mapping[num] for num in encoded_labels]
    return labels


def prep_data(data, metadata, save_weights=False, test_frac=0.2):
    """
    Prep raw data for models.

    Parameters
    ----------
    data : df
        Time series data.
    metadata : df
        Corresponding metadata.

    Returns
    -------
    dataset : TensorDataset
        torch dataset prepped for DataLoader.

    """
    
    # Normalize X
    data_norm = normalize_data(data)
    
    # Encode y
    states_array = metadata["state"].to_numpy()
    states_encoded = labels_to_numbers(states_array)
    
    # Print encoding and counts
    df_summary = pd.DataFrame({
        "State": states_array,
        "Encoded": states_encoded
    })
    summary_table = df_summary.groupby(["State", "Encoded"]).size().reset_index(name="Count")
    summary_table = summary_table.sort_values("Encoded")  # optional, sorts by encoded number
    print("\nState encoding summary:")
    print(summary_table.to_string(index=False))
    
    # Create full TensorDataset
    full_dataset = TensorDataset(
        torch.tensor(data_norm, dtype=torch.float32).unsqueeze(1),
        torch.tensor(states_encoded, dtype=torch.long)
    )
    
    # Compute split sizes
    total_len = len(full_dataset)
    test_len = int(total_len * test_frac)
    train_len = total_len - test_len
    
    # Split dataset and save indices to reload
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])
    
    
    # Save weights if models have not been run for reloading/analysis
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    if save_weights == True:
        torch.save(train_dataset.indices, os.path.join(models_dir, "train_dataset_indices.pth"))
        torch.save(test_dataset.indices, os.path.join(models_dir, "test_dataset_indices.pth"))
    
    return full_dataset, train_dataset, test_dataset


def plot_timeseries(data, metadata):
    """

    Parameters
    ----------
    data : df
        Time series data.
    metadata : df
        Corresponding metadata.

    Returns
    -------
    None. Plots time series data.

    """
    
    # Find indices where recording number changes to pull one time series from each recording
    change_indices = metadata.index[metadata["filename"].shift() != metadata["filename"]]
    idxs = list(change_indices)
    
    # Find corresponding recording numbers and states
    fnames = metadata.loc[idxs, "filename"].tolist()
    statenames = metadata.loc[idxs, "state"].tolist()
    recnames = [(x.split('_')[1]) for x in fnames]
    
    # Plot one instance of time series data from each of the 13 recordings
    plt.figure(figsize=(12, 20))
    for i, (idx, fname, state) in enumerate(zip(idxs, recnames, statenames)):
        plt.subplot(13, 1, i + 1)
        plt.plot(data[idx])
        plt.title(f"{fname}: {state}")
        plt.xlabel("Time Index (ms)")
        plt.ylabel(r"Microvolts ($\mu$V)")
    
    plt.tight_layout(h_pad=2.0)
    plt.show()
    
    
def plot_timeseries_per_state(data, metadata):
    """
    Plot one time series per state class, keeping recording number formatting.

    Parameters
    ----------
    data : df
        Time series data.
    metadata : df
        Corresponding metadata.

    Returns
    -------
    None. Plots time series data.
    """
    # Track which states we have already plotted
    plotted_states = set()

    plt.figure(figsize=(12, 3 * len(metadata["state"].unique())))  # height scales with number of classes

    for idx, row in metadata.iterrows():
        state = row["state"]
        if state not in plotted_states:
            fname = row["filename"]
            recnum = fname.split('_')[1]  # Keep original recording number formatting

            plt.subplot(len(metadata["state"].unique()), 1, len(plotted_states) + 1)
            plt.plot(data[idx])
            plt.title(f"{recnum}: {state}")
            plt.xlabel("Time Index (ms)")
            plt.ylabel(r"Microvolts ($\mu$V)")

            plotted_states.add(state)

        # Stop once we've plotted one example for each state
        if len(plotted_states) == len(metadata["state"].unique()):
            break

    plt.tight_layout(h_pad=2.0)
    plt.show()
        