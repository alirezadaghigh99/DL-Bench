#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the PFRL repository
echo "Cloning the PFRL repository..."
git clone https://github.com/pfnet/pfrl.git
cd pfrl

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch>=1.3.0 gym>=0.9.7 numpy>=1.10.4 filelock pillow

# Install PFRL from the source
echo "Installing PFRL from source..."
python setup.py install

echo "PFRL installation complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
