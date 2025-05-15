#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the Pyro repository
echo "Cloning the DeepReg repository..."
git clone https://github.com/DeepRegNet/DeepReg.git
cd DeepReg



# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate
pip install pytest

# Install Pyro
echo "Installing DeepReg..."
pip install -e .
pip install pytest
echo "Pyro installation complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
