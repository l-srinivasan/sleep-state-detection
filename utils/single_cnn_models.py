#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 2 18:20:07 2025
@author: Leela Srinivasan
"""


import torch.nn as nn


class CNN(nn.Module):
    """
    Original CNN created by the authors."""
    
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            # They specify dims, so this will error if data != 2000 timesteps
            nn.Linear(128 * 250, 256),  # Adjusted based on pooling and data dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
class ModifiedCNN(nn.Module):
    """
    My modified CNN with the following changes:
    (1) Global Average Pooling
    (2) A better receptive field with larger kernels
    (3) Batch Normalization
    (4) Smaller FC/Dense layers to reduce parameters
    (5) Dropout
    (6) Adaptive Pooling in the case of variable length time series data."""
    
    def __init__(self):
        super(ModifiedCNN, self).__init__()

        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            # Add FLatten/Linear layers
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4)      
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x