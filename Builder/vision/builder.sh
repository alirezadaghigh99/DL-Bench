#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the torchvision repository
echo "Cloning the torchvision repository..."
# git clone https://github.com/pytorch/vision.git
cd vision

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3 -m venv venv  # Ensure Python 3 or higher is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install PyTorch (required for building torchvision)
echo "Installing PyTorch..."
pip install torch  # You may specify the version as needed (e.g., torch==1.x.x)

# Install development dependencies
echo "Installing development dependencies..."
pip install expecttest flake8 typing mypy pytest pytest-mock scipy requests

# Build and install torchvision in development mode
echo "Building and installing torchvision in development mode..."
python setup.py develop  # Replace with 'install' if development mode is not required

# For macOS setup
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Configuring for macOS build..."
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CC=clang
  export CXX=clang++
  python setup.py develop  # Adjust with 'install' if needed
fi

# For C++ debugging
# Uncomment the following line to enable debugging:
# DEBUG=1 python setup.py develop

# Optional: Force CUDA build
# Uncomment the following line to force GPU support if needed:
# export FORCE_CUDA=1

echo "Torchvision setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
