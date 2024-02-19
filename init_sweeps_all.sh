#!/bin/bash

# GENERATE SWEEPS USING WANDB API

# wandb login information (change the following to your own) 
WANDB_API_KEY=your_wandb_api_key # This is secret and shouldn't be checked into version control
PROJECT=your_wandb_project_name # wandb project name
ENTITY=your_wandb_username # wandb username

# ----------------------- DON'T CHANGE BELOW ------------------------------
# Generate sweeps and save sweep IDs into a text file
for file in sweep_config_files/*.yaml;
    do
        CONFIG_FILE=$file

        # Create text file to store sweep ids
        touch sweepid_all.txt

        # Create sweep
        wandb sweep --project $PROJECT $CONFIG_FILE > sweepid_tmp.txt 2>&1

        # Extract the sweep ID using awk
        SWEEP_ID=$(awk '/wandb: Created sweep with ID:/ {print $6}' sweepid_tmp.txt)
        rm sweepid_tmp.txt
        echo "$CONFIG_FILE - $SWEEP_ID" >> sweepid_all.txt
done;
