# Replicative study: Neural models for detection and classification of brain states and transitions

## Overview
This project is a comparative study of the Nature Communications Biology paper "Neural models for detection and classification of brain states and transitions" by authors Arnau Marin-Llobet et al. The project aims to reproduce and expand upon the results reported by the authors in the study of classifying anesthesia-induced sleep states and transitions between them.

## Data Access
The dataset must be downloaded directly from Zenodo: https://zenodo.org/records/14990181 and placed in the folder /data with the default name data_states.npy for easy interaction with the python notebook tutorials. The metadata file metadata_states.pkl containing ground truth labeling is stored under /data.

## Usage
There are two principle python notebook tutorials in /notebooks to walk users through the described methods. Users can run the notebooks using Google Colaboratory on CPUs for convenience.

Tutorial 1: classify_states.ipynb

Tutorial 2: detect_transitions.ipynb

All models are currently trained and evaluated with saved results to encourage user flexibility. The indices from the training/testing split have been saved if the user wishes to directly reproduce the reported results. If the user wishes to run everything from scratch, they should rename the /models directory to avoid overwriting saved models and results.

## Repository Tree
```
sleep-state-detection/
├── author_autoencoders
│   ├── autoencoder_simpl_state_asynch_MA_epochs_150.h5
│   ├── autoencoder_simpl_state_awake_epochs_150.h5
│   ├── autoencoder_simpl_state_slow_MA_epochs_150.h5
│   └── autoencoder_simpl_state_slow_updown_epochs_150.h5
├── data
│   ├── metadata_states.pkl
│   └── metadata.parquet
├── img
│   ├── ae_reconstruction.png
│   ├── arch_models_1_2.png
│   ├── cm_model_1.png
│   ├── cm_model_2.png
│   ├── cm_model_3.png
│   ├── detected_transition.png
│   ├── fc_model_3.png
│   ├── metrics_across_thresholds.png
│   ├── psd_ellipses.png
│   ├── psd_scatter.png
│   ├── train_val_acc.png
│   ├── transition_3by3.png
│   └── unknown_percentages.png
├── models
│   ├── cnn_modified_optimized_0.5
│   │   ├── all_preds_df_cnn_modified_optimized_0.5.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_modified_optimized_0.5.pkl
│   │   └── unknown_df_cnn_modified_optimized_0.5.pkl
│   ├── cnn_modified_optimized_0.6
│   │   ├── all_preds_df_cnn_modified_optimized_0.6.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_modified_optimized_0.6.pkl
│   │   └── unknown_df_cnn_modified_optimized_0.6.pkl
│   ├── cnn_modified_optimized_0.7
│   │   ├── all_preds_df_cnn_modified_optimized_0.7.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_modified_optimized_0.7.pkl
│   │   └── unknown_df_cnn_modified_optimized_0.7.pkl
│   ├── cnn_modified_optimized_0.8
│   │   ├── all_preds_df_cnn_modified_optimized_0.8.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_modified_optimized_0.8.pkl
│   │   └── unknown_df_cnn_modified_optimized_0.8.pkl
│   ├── cnn_modified_optimized_0.9
│   │   ├── all_preds_df_cnn_modified_optimized_0.9.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_modified_optimized_0.9.pkl
│   │   └── unknown_df_cnn_modified_optimized_0.9.pkl
│   ├── cnn_modified_optimized_training
│   │   ├── cnn_modified_optimized_metrics.json
│   │   ├── model_cnn_modified_optimized_epoch10.pth
│   │   └── model_cnn_modified_optimized_epoch5.pth
│   ├── cnn_standard_default_0.9
│   │   ├── all_preds_df_cnn_standard_default_0.9.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_standard_default_0.9.pkl
│   │   └── unknown_df_cnn_standard_default_0.9.pkl
│   ├── cnn_standard_default_training
│   │   ├── cnn_standard_default_metrics.json
│   │   ├── model_cnn_standard_default_epoch10.pth
│   │   └── model_cnn_standard_default_epoch5.pth
│   ├── cnn_standard_optimized_0.9
│   │   ├── all_preds_df_cnn_standard_optimized_0.9.pkl
│   │   ├── confusion_matrix.png
│   │   ├── metrics_df_cnn_standard_optimized_0.9.pkl
│   │   └── unknown_df_cnn_standard_optimized_0.9.pkl
│   ├── cnn_standard_optimized_training
│   │   ├── cnn_standard_optimized_metrics.json
│   │   ├── model_cnn_standard_optimized_epoch10.pth
│   │   └── model_cnn_standard_optimized_epoch5.pth
│   ├── test_dataset_indices.pth
│   └── train_dataset_indices.pth
├── notebooks
│   ├── classify_states.ipynb
│   ├── detect_transitions.ipynb
│   └── supplementary.ipynb
├── README.md
├── ref.pdf
├── report.pdf
└── utils
    ├── autoencoder_models.py
    ├── autoencoder_utils.py
    ├── dual_cnn_models.py
    ├── eval.py
    ├── preprocess.py
    ├── single_cnn_models.py
    └── train.py
```

## Credit
Sub-blocks of code have been pulled directly from the authors' publicly available codebase (https://github.com/arnaumarin/LFPDeepStates/tree/main) in attempts to recreate methodology. Code has been revised for reuse but some lines may be identical, particularly in architecture formation and model evaluation.
