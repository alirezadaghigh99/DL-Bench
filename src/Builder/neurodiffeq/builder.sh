#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the neurodiffeq repository
echo "Cloning the PFRL repository..."
git clone  https://github.com/NeuroDiffGym/neurodiffeq.git && cd neurodiffeq

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch>=1.3.0 gym>=0.9.7 numpy>=1.10.4 filelock pillow

# Install neurodiffeq from the source
echo "Installing neurodiffeq from source..."
pip install -e .

echo "neurodiffeq installation complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
