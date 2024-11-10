#!/bin/bash

# Exit the script if any command fails
set -e
# Clone the inference repository
echo "Cloning the inference repository..."
# git clone https://github.com/roboflow/inference.git
cd inference
# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3.10 -m venv venv  # Ensure Python 3.10 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install the current repository in editable mode
echo "Installing the repository in editable mode..."
pip install -e .

echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
