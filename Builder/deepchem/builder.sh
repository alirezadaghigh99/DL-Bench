#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the DeepChem repository
echo "Cloning the DeepChem repository..."
# git clone https://github.com/deepchem/deepchem.git
cd deepchem

# If DeepChem already exists, update it

# Create a Python virtual environment
echo "Creating a virtual environment..."
python3.9 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate

# Install deep learning frameworks (modify as needed)
echo "Installing TensorFlow..."
pip install tensorflow

# Optionally install PyTorch or JAX (uncomment as needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# echo "Installing JAX..."
pip install jax jaxlib
pip install --upgrade pip setuptools

# Install DeepChem in develop mode
echo "Installing DeepChem in develop mode..."
# pip install -r requirements.txt  # Install dependencies
pip install -e .

echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
