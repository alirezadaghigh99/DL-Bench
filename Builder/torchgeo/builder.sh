#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the TorchGeo repository
echo "Cloning the TorchGeo repository..."
# git clone https://github.com/microsoft/torchgeo.git
cd torchgeo

# Create a Python virtual environment
echo "Creating a virtual environment..."
python3.10 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate

# Check if Poetry is required for installation
if grep -q "\[tool.poetry\]" pyproject.toml; then
  echo "Poetry detected. Installing poetry..."
  pip install poetry
  echo "Installing dependencies with poetry..."
  poetry install
else
  # Install dependencies and the package using pip
  echo "Installing dependencies with pip..."
  pip install .
fi

echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
