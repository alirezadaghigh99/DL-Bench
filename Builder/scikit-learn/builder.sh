#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the scikit-learn repository
echo "Cloning the scikit-learn repository..."
git clone https://github.com/scikit-learn/scikit-learn.git  # add --depth 1 for a shallow clone if needed
cd scikit-learn

# Create a Python virtual environment
echo "Creating a virtual environment..."
python3.9 -m venv venv  # Ensure Python 3.9 or later is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip to a compatible version
echo "Upgrading pip..."
pip install --upgrade pip

# Install build and runtime dependencies
echo "Installing build and runtime dependencies..."
pip install wheel numpy scipy cython meson-python ninja

# Install scikit-learn in develop mode
echo "Building and installing scikit-learn in develop mode..."
pip install --editable . \
   --verbose --no-build-isolation \
   --config-settings editable-verbose=true

# Verify the installation
echo "Verifying the scikit-learn installation..."
python -c "import sklearn; sklearn.show_versions()"
pip install pytest
echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
