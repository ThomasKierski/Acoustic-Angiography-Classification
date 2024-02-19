#!/bin/bash

# TRAINS THE FINAL MODELS

# wandb login information (change the following to your own) 
WANDB_API_KEY=your_wandb_api_key # This is secret and shouldn't be checked into version control
export PROJECT=your_wandb_project_name # wandb project name
ENTITY=your_wandb_username # wandb username

# Define path to directories
export DATA_DIR='/path/to/zenodo_data_directory'
export SCRIPT_DIR='/path/to/github_repository'

# Select network to run (comment/uncomment accordingly)
NETWORK=EfficientNetB0
# NETWORK=EfficientNetB1
# NETWORK=DenseNet121
# NETWORK=ResNet

# Select number of spatial dimensions (comment/uncomment accordingly)
# SPATIAL_DIMS = 3 # only for EfficientNetB0
SPATIAL_DIMS=2 # any network

# ----------------------- DON'T CHANGE BELOW ------------------------------
echo "DATA_DIR: $DATA_DIR"

# Data paths
export DATA3D_PATH="$DATA_DIR/FOR_TRAINING/3d_data.npz"
export DATA2D_PATH="$DATA_DIR/FOR_TRAINING/2d_data.npz"

wandb login $WANDB_API_KEY; 
cd $SCRIPT_DIR; 
python3 -m aa_classification_nested_kfold_final_models.py --spatial_dims=$SPATIAL_DIMS --network=$NETWORK
