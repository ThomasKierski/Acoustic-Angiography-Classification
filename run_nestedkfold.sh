#!/bin/bash

# GENERATE AND TRACK NESTED K-FOLD EXPERIMENT USING WANDB API

# wandb login information (change the following to your own) 
WANDB_API_KEY=your_wandb_api_key # This is secret and shouldn't be checked into version control
PROJECT=your_wandb_project_name # wandb project name
ENTITY=your_wandb_username # wandb username

# Select which sweep to run by commenting/uncommenting accordingly

# CONFIG_FILE=DenseNet121_config.yaml 
CONFIG_FILE=ENB0_config.yaml 
# CONFIG_FILE=ENB1_config.yaml 
# CONFIG_FILE=ResNet18_config.yaml 
# CONFIG_FILE=ENB03D_config.yaml 

# Define path to data directory
export DATA_DIR='/path/to/zenodo_data_directory'
export SCRIPT_DIR='/path/to/github_repository'

# ----------------------- DON'T CHANGE BELOW ------------------------------
echo "DATA_DIR: $DATA_DIR"

# Data paths
export DATA3D_PATH="$DATA_DIR/FOR_TRAINING/3d_data.npz"
export DATA2D_PATH="$DATA_DIR/FOR_TRAINING/2d_data.npz"

# Extract  desired the sweep ID (from sweepid_all.txt) using awk
SWEEP_ID=$(awk '/'"$CONFIG_FILE"'/ {print $3}' sweepid_all.txt)
echo "SWEEP_ID: $SWEEP_ID"
echo "$CONFIG_FILE"

cd $SCRIPT_DIR
wandb login $WANDB_API_KEY
wandb agent $ENTITY/$PROJECT/$SWEEP_ID
