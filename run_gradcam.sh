#!/bin/bash

# PERFORM GRADCAM ANALYSIS ON 2D DATA

# Define path to directories
export GRADCAM_DIR='/path/to/gradcam_results_directory'
export DATA_DIR='/path/to/zenodo_data_directory'

# ----------------------- DON'T CHANGE BELOW ------------------------------
export MODEL_DIR="$DATA_DIR/NESTED_KFOLD/trained_models/2d_models"
export DATA2D_PATH="$DATA_DIR/FOR_TRAINING/2d_data.npz"

mkdir $GRADCAM_DIR;
mkdir $GRADCAM_DIR/TP;
mkdir $GRADCAM_DIR/TN;
mkdir $GRADCAM_DIR/FN;
mkdir $GRADCAM_DIR/FP;

python3 -m run_gradcam_analysis