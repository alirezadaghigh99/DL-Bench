#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the gpflow repository
echo "Cloning the ignite repository..."
git clone https://github.com/pytorch/ignite.git
cd ignite

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate



# Install ignite as an editable package
echo "Installing ignite as an editable package..."
pip install -e .
pip install pytest
pip install torchvision
echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
