#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the imagededup repository
echo "Cloning the imagededup repository..."
git clone https://github.com/idealo/imagededup.git
cd imagededup

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install Cython
echo "Installing Cython..."
pip install "cython>=0.29"

# Install imagededup
echo "Installing imagededup from source..."
python setup.py install

echo "Installation complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
