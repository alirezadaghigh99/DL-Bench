#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the stanza repository
echo "Cloning the stanza repository..."
git clone https://github.com/stanfordnlp/stanza.git
cd stanza

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate



# Install cleanlab as an editable package
echo "Installing stanza as an editable package..."
pip install -e .
pip install pytest

echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
