#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:41:34 2025
@author: Leela Srinivasan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared feature extractor
class CNN_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        return self.features(x)


# Level 1: Classify Awake, Slow Up-Down, and Microarousals
class CNN_Level1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CNN_Base()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 250, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)   # 3 classes
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Level 2: Binary Classification to subdivide Microarousals
class CNN_Level2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CNN_Base()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 250, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Hierarchical Classifier to merge CNN_Level1, CNN_Level2
class HierarchicalCNN(nn.Module):
    """
    Produces final four classes
    """

    def __init__(self, model1, model2):
        super().__init__()
        self.level1 = model1
        self.level2 = model2

    def forward(self, x, return_logits=False):
        """
        If return_logits=False, returns final class indices
        If return_logits=True, returns a 4-class probability vector
        """

        out1 = F.softmax(self.level1(x), dim=1)
        pred1 = torch.argmax(out1, dim=1)

        # Prepare output container
        batch_size = x.size(0)

        if not return_logits:
            # Final class index for each batch element
            final_pred = torch.zeros(batch_size, dtype=torch.long, device=x.device)

            for i in range(batch_size):
                if pred1[i] == 0:
                    final_pred[i] = 0
                elif pred1[i] == 1:
                    final_pred[i] = 1
                else:
                    # Run Level-2 only for C
                    out2 = F.softmax(self.level2(x[i].unsqueeze(0)), dim=1)
                    pred2 = torch.argmax(out2, dim=1).item()
                    final_pred[i] = 2 + pred2  # 2 or 3
            return final_pred

        else:

            # Return combined four class logits
            probs = torch.zeros(batch_size, 4, device=x.device)

            for i in range(batch_size):
                if pred1[i] == 0:
                    probs[i, 0] = out1[i, 0]
                elif pred1[i] == 1:
                    probs[i, 1] = out1[i, 1]
                else:
                    out2 = F.softmax(self.level2(x[i].unsqueeze(0)), dim=1)
                    probs[i, 2] = out2[0, 0] 
                    probs[i, 3] = out2[0, 1]
            return probs
