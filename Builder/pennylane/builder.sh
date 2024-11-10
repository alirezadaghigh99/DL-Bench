#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the PennyLane repository
echo "Cloning the PennyLane repository..."
git clone https://github.com/PennyLaneAI/pennylane.git
cd pennylane

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install PennyLane in editable mode
echo "Installing PennyLane in editable mode..."
pip install -e .

# Install additional development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

echo "PennyLane development setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
