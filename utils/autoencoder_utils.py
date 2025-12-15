#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 9 17:53:55 2025
@author: Leela Srinivasan
"""

# Import computational modules
import numpy as np
from scipy.signal import welch
import pandas as pd


# Import plotting modules
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Ellipse


from utils.preprocess import numbers_to_labels


def print_reconstruction_loss(loss_dict, state_name):
    """
    Print autoencoder reconstruction loss for a given state

    Parameters
    ----------
    loss_dict : dict
        dict containing loss.
    state_name : str
        State label, not encoding.

    Returns
    -------
    None. Prints.

    """
    print(f"\nReconstruction Loss for {state_name} Test Sample")
    for k, v in loss_dict.items():
        print(f"{k:25s} : {float(v):.6f}")

    predicted = min(loss_dict, key=loss_dict.get)
    print(f"Predicted Classification: {predicted}")
    print("\n")
    

def reconstruction_loss(samples, states, autoencoders):
    """
    Sister function to print_reconstruction_loss above.

    Parameters
    ----------
    samples : array
        data for several samples.
    states : str
        state name, not encoding.
    autoencoders : models
        saved/trained autoencoders to run predictions.

    Returns
    -------
    None. Prints.

    """
    
    for sample, state in zip(samples, states):
        errors = {}
        for name, ae in autoencoders.items():
            sample = np.array(sample).reshape(1, -1)
            output = ae.predict(sample)
            err = np.mean((sample - output)**2)
            errors[name] = err
          
        print_reconstruction_loss(errors, state)
        
        
def prep_unknown_sample_df(f):
    """
    Prep the df containing unknown/unclassified sample info

    Parameters
    ----------
    f : str
        path to df of unknown samples.

    Returns
    -------
    df : df
        df with encodings converted to labels.

    """
    
    df = pd.read_pickle(f)
    df = df.reset_index(drop=True)
    df["state"] = df["y_true"].apply(lambda x: numbers_to_labels([x])[0])
    df.head()
    return df


def gen_reconstruction(df, test_data, autoencoders):
    """
    Use autoencoder to generate econstructed samples.

    Parameters
    ----------
    df : df
        df of unknown sampples.
    test_data : df
        normalized test data, not in TensorDataset format.
    autoencoders : autoencoder models
        trained autoencoders.

    Returns
    -------
    x_hat : array
        reconstructed data.

    """
    
    row = df.iloc[0]
    sample = test_data[row["index"]]
    sample = np.asarray(sample).reshape(1, -1)
    ae_type = row["state"]
    x_hat = autoencoders[ae_type].predict(sample)
    
    plt.figure(figsize=(12, 4))
    plt.plot(sample.squeeze())
    plt.plot(x_hat.squeeze())
    plt.title("Autoencoder Time Series Reconstruction")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.legend(["Original Sample", "Reconstruction"])
    plt.show()
    
    return x_hat


def band_power_from_psd(f, Pxx, band):
    """
    Compute band-limited power by integrating PSD over specified frequency range.
    
    Parameters
    ----------
    f : array
        freq corresponding to the power spectral density estimates.
    Pxx : array
        power spectral density (PSD) values returned by Welch's method.
    band : tuple of float
        freq band specified as (low, high) in Hz.

    Returns
    -------
    float
        band power.

    """
    low, high = band
    mask = (f >= low) & (f <= high)
    Pxx = np.squeeze(Pxx)
    if not np.any(mask):
        return 0.0
    return np.trapz(Pxx[mask], f[mask])


def compute_band_powers(x, nfft=16384):
    """
    Compute PSD using Welch's method and extract band-limited power features.

    Parameters
    ----------
    x : array
        1D time-series signal.
    nfft : int, optional
        Number of FFT points used in Welch's method. The default is 16384.

    Returns
    -------
    powers : dict
        contains band power values for each frequency band
        ('delta', 'theta', 'gamma') and total power.

    """
    # Set sampling rate and PSD frequency bands
    fs = 1000
    bands = {
        "delta": (0.1, 1),
        "theta": (4, 8),
        "gamma": (100, 500),
    }

    # Compute PSD with Welch, using a large nfft for finer interpolation of bins
    nperseg = 256
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, nfft=nfft, scaling="density", window="hann")

    powers = {}
    for name, band in bands.items():
        powers[name] = band_power_from_psd(f, Pxx, band)

    # Total power across all frequencies
    powers["total_power"] = np.trapz(Pxx, f)
    return powers, (f, Pxx)


def append_psd_bands(df, test_data, autoencoders):
    """
    Add PSD band information to unknown sample df

    Parameters
    ----------
    df : df
        unkknown sample df.
    test_data : 2d array
        numpy array of normalized test data, not in TensorDataset format.
    autoencoders : model class
        pretrained autoencoders.

    Returns
    -------
    df : df
        df with updated PSD info.

    """
    
    # Prepare lists to hold the power values
    delta_powers = []
    theta_powers = []
    gamma_powers = []
    
    for _, row in df.iterrows():
        # Retrieve original signal
        sample = test_data[row["index"]]
        sample = np.asarray(sample).reshape(1, -1) # Reshape for autoencoder
    
        # Choose autoencoder based on state with maximal CNN confidence
        ae_type = row["state"]
        x_hat = autoencoders[ae_type].predict(sample, verbose=0).squeeze()  # shape (2000,)
    
        # Compute band powers
        powers, (f, Pxx) = compute_band_powers(x_hat)
    
        # Append to lists
        delta_powers.append(powers["delta"])
        theta_powers.append(powers["theta"])
        gamma_powers.append(powers["gamma"])
    
    # Append columns to df
    df["delta_power"] = delta_powers
    df["theta_power"] = theta_powers
    df["gamma_power"] = gamma_powers
    return df


def euclidean(a, b):
    """
    Compute euclidean between a and b

    Parameters
    ----------
    a : array
        3d coord.
    b : array
        3d coord.

    Returns
    -------
    float
        distance.

    """
    return np.linalg.norm(a - b)


def compute_centroids(df):
    """
    Compute the centroids for each class. Compute the transition centroid as the midpoint between
    microarousal centroids, and see whether each sample is closer to the transition centroid
    than it's own centroid.

    Parameters
    ----------
    df : df
        unknown sample df.

    Returns
    -------
    df : df
        unknown sample df updated with transition info.

    """
    
    features = ["delta_power", "theta_power", "gamma_power"]
    centroids = {}
    for cls in sorted(df["y_true"].unique()):
        centroids[cls] = df[df["y_true"] == cls][features].mean().values
    
    # Compute "transition" centroid: midpoint between class 2 and 3 centroids
    transition_centroid = (centroids[2] + centroids[3]) / 2

    # Prepare array form
    coords = df[features].values
    classes = df["y_true"].values
    
    # Compute distance to own centroid
    dist_own = np.array([euclidean(coords[i], centroids[classes[i]]) for i in range(len(df))])
    
    # Distance to transition centroid
    dist_transition = np.linalg.norm(coords - transition_centroid, axis=1)
    
    # Check which is closer
    df["closer_to_transition"] = dist_transition < dist_own
    return df


def plot_transitions_3by3(df, test_data):
    """
    Plot 9 example transition samples

    Parameters
    ----------
    df : df
        df with updated transition info.
    test_data : 2d array
        testing data, normalized.

    Returns
    -------
    None. Plots.

    """
    # Plot 9 detected transition samples to get a feel for morphology

    df_sub = df[(df["closer_to_transition"] == True) & (df["y_true"].isin([0, 2, 3]))]
    
    samples_0 = df_sub[df_sub["y_true"] == 0].head(3)
    samples_2 = df_sub[df_sub["y_true"] == 2].head(3)
    samples_3 = df_sub[df_sub["y_true"] == 3].head(3)
    
    # Combine in desired order: 0, 2, 3
    all_samples = pd.concat([samples_0, samples_2, samples_3])
    
    # Create 3 by 3 plot
    plt.figure(figsize=(14, 10))
    
    for i, (_, row) in enumerate(all_samples.iterrows()):
        idx = row["index"]
        y = test_data[idx]
        
        plt.subplot(3, 3, i + 1)
        plt.plot(y, linewidth=1)
        plt.title(f"Labeled Class {row['y_true']}  |  idx={idx}")
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle("Detected Transitions across Labeled Classes", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


def plot_transition_sample(data, test_data, transition_idx, asynch_ma_idx, slow_ma_idx):
    """
    Plot a transition sample. Must select index for transition sample, example asynch MA, and example slow MA samples.

    Parameters
    ----------
    data : np array
        full data.
    test_data : np array
        test data.
    transition_idx : int
        index from test data
    asynch_ma_idx : TYPE
        index from data.
    slow_ma_idx : TYPE
        index from data.

    Returns
    -------
    None. Plots.

    """
    # Plot transition sample against cleaner samples from each class
    
    # Pull transition idx, asynch MA idx and slow MA idx
    idx1, idx2, idx3 = transition_idx, asynch_ma_idx, slow_ma_idx
    plt.figure(figsize=(15, 4))
    
    # Subplot 1
    plt.subplot(1, 3, 1)
    plt.plot(test_data[idx1])
    plt.title("Detected Transition Sample")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    
    # Subplot 2
    plt.subplot(1, 3, 2)
    plt.plot(data[idx3])
    plt.title("Slow MA Sample")
    plt.xlabel("Time (ms)")
    
    # Subplot 3
    plt.subplot(1, 3, 3)
    plt.plot(data[idx2])
    plt.title("Asynch MA Sample")
    plt.xlabel("Time (ms)")
    
    plt.tight_layout()
    plt.show()
    

def build_colormap(df):
    """
    Build the color mapping for the 3d PSD plots.

    Parameters
    ----------
    df : df
        df of unknown samples with PSD bands updated .

    Returns
    -------
    df : df
        updated with color mpaping.
    color_map : dict
        color mapping.

    """
    
    color_map = {
        "Awake": "black",
        "Slow Up-Down": "green",
        "Slow MA": "red",
        "Asynch MA": "blue",
        "Transition": "orange"
    }
    
    
    df["det_state"] = np.where(df["closer_to_transition"], "Transition", numbers_to_labels(df["y_true"]))
    df["plotting_color"] = df["det_state"].map(color_map)
    return df, color_map


def plot_psd(df, color_map):
    """
    Plot PSD color coded by state

    Parameters
    ----------
    df : df
        df with PSD band info.
    color_map : dict
        color mapping.

    Returns
    -------
    None. Plots.

    """
    
    # Plot PSD for Detected Classes

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df["delta_power"],
        df["theta_power"],
        df["gamma_power"],
        c=df["plotting_color"],
        s=10,
        alpha=0.8
    )
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.set_xlabel('Delta Power (0.1–4 Hz)')
    ax.set_ylabel('Theta Power (4–8 Hz)')
    ax.set_zlabel('Gamma Power (100–500 Hz)')
    ax.set_title('PSD Band Powers by Detected State')
    
    # Custom legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=state,
            markerfacecolor=color_map[state],
            markersize=8
        )
        for state in color_map
    ]
    
    ax.legend(handles=legend_elements, title="Detected State", loc='upper right')
    
    ax.view_init(elev=20, azim=40)
    plt.tight_layout()
    plt.show()


def plot_psd_ellipses(df, color_map):
    """
    Same as above, with visual ellipses

    Parameters
    ----------
    df : df
        df of unknown samples with PSD info.
    color_map : dict
        color mapping.

    Returns
    -------
    None. Plots.

    """

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter points
    ax.scatter(
        df["delta_power"],
        df["theta_power"],
        df["gamma_power"],
        c=df["plotting_color"],
        s=10,
        alpha=0.8
    )
    
    # For each state, add an ellipse on the x-y plane
    for state, color in color_map.items():
        group = df[df["det_state"] == state]
        if len(group) == 0:
            continue
            
        # Compute mean and std in x,y for ellipse
        cx = group["delta_power"].mean()
        cy = group["theta_power"].mean()
        sx = group["delta_power"].std()
        sy = group["theta_power"].std()
        
        # Ellipse patch (2 std dev radius)
        ellipse = Ellipse(
            (cx, cy),
            width=4*sx,
            height=4*sy,
            facecolor=color,
            edgecolor='none',
            alpha=0.15   # light shading
        )
        
        # Add to 3D plot (z=constant)
        ax.add_patch(ellipse)
        art3d.pathpatch_2d_to_3d(ellipse, z=group["gamma_power"].mean(), zdir="z")
    
    # Labels
    ax.set_xlabel("Delta Power")
    ax.set_ylabel("Theta Power")
    ax.set_zlabel("Gamma Power")
    ax.set_title("PSD Band Powers with Cluster Shading")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label=state,
               markerfacecolor=color_map[state], markersize=8)
        for state in color_map
    ]
    ax.legend(handles=legend_elements, title="Detected State", loc='upper right')
    
    ax.view_init(elev=20, azim=40)
    plt.tight_layout()
    plt.show()
