#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the nncf repository
echo "Cloning the nncf repository..."
# git clone https://github.com/openvinotoolkit/nncf.git
cd nncf

# Set up a virtual environment
echo "Creating a virtual environment..."
python3.9 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate

# Install dependencies
echo "Installing development requirements..."
# pip install -r requirements.txt

# Install nncf as an editable package
echo "Installing nncf as an editable package..."
pip install .
pip install tensorflow==2.17.0 torch onnx pytest-mock addict requests torchvision efficientnet_pytorch pytest_dependency tensorboard transformers datasets
echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
