#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the small-text repository
echo "Cloning the small-text repository..."
git clone https://github.com/webis-de/small-text.git
cd small-text

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate

# Install dependencies
echo "Installing development requirements..."
pip install -r requirements-dev.txt

# Install small-text as an editable package
echo "Installing small-text as an editable package..."
pip install -e .
pip install pytest torch
# If running on a Mac with Apple silicon, replace TensorFlow packages


echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
