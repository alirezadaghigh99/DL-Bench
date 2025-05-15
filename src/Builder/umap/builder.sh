#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the umap repository
echo "Cloning the umap repository..."
# git clone https://github.com/lmcinnes/umap.git
cd umap

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate



# Install cleanlab as an editable package
echo "Installing umap as an editable package..."
pip install -e .


echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
