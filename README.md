# AA-CNN-Classification

## Requirements

The file requirements.txt lists the Python packages (and specific versions) required to run the scripts defined below.

Accompanying data can be found in the following Zenodo repository: (Zenodo DOI link will be added once available)

## Overview

Acoustic angiography is a contrast-enhanced ultrasound imaging modality that leverages the non-linear response of ultrasound contrast agents (also known as microbubbles) to generate high resolution and high contrast images of microvasculature with minimal tissue background[1]. Using the bash and Python scripts in this repository, we have trained convolutional neural networks (EfficientNet-B0, EfficientNet-B1, DenseNet-121, ResNet-18) to classify between acoustic angiography images (2-D) and volumes (3-D) of tumor-bearing and healthy tissue acquired in vivo in a nested k-fold cross validation study, optimizing hyperparameters on the inner folds and evaluating model performance on the outer folds. The datasets were packaged for training and are available in the Zenodo repository linked above. We utilized the WandB platform to track model training [2]. 

## Usage

Run the bash scripts in the following order:

1. init_sweeps_all.sh - initiates a hyperparameter sweep using the WandB platform based on the sweep configurations defined in the sweep_yaml_files folder and outputs a file (sweepid_all.txt) containing the sweep id
    - Change the WANDB_API_KEY (WandB api key), PROJECT (WandB project), and ENTITY (WandB username) variable definitions

2. run_nestedkfold.sh - initiates a nested k-fold cross validation study for a selected network by running aa_classification_nested_kfold.py, tracked through WandB
    - Change the WANDB_API_KEY (WandB api key), PROJECT (WandB project), and ENTITY (WandB username) variable definitions
    - Change the DATA_DIR and SCRIPT_DIR paths to point to the Zenodo data directory and this GitHub directory, respectively
    - Comment/uncomment the correct CONFIG_FILE line to run the desired network

3. train_final_models.sh - loads the model configuration with the best loss for a selected network and trains the model on outerfold data by running aa_classification_nested_kfold_final_models.py, tracked through WandB
    - Change the WANDB_API_KEY (WandB api key), PROJECT (WandB project), and ENTITY (WandB username) variable definitions
    - Change the DATA_DIR and SCRIPT_DIR paths to point to the Zenodo data directory and this GitHub directory, respectively
    - Comment/uncomment the correct NETWORK and SPATIAL_DIMS lines to run the desired network

4. run_gradcam.sh - performs gradient-weighted class activation mapping (GradCAM) on the 2-D dataset by running run_gradcam_analysis.py and outputs GradCAM saliency maps in a defined GradCAM directory
    - Change the DATA_DIR and GRADCAM_DIR paths to point to the Zenodo data directory and a directory for saving GradCAM results, respectively

## References

[1] R. C. Gessner, C. B. Frederick, F. S. Foster, and P. A. Dayton, “Acoustic Angiography: A New Imaging Modality for Assessing Microvasculature Architecture,” International Journal of Biomedical Imaging, vol. 2013, p. e936593, Jul. 2013, doi: 10.1155/2013/936593.
[2] L. Biewald, “Experiment tracking with weights and biases.” 2020. [Online]. Available: https://www.wandb.com/

## License

The codes are licensed under the MIT license.
