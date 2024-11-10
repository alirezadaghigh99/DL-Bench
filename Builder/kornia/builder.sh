#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the Kornia repository from your GitHub
echo "Cloning the Kornia repository..."
# git clone https://github.com/kornia/kornia.git
cd kornia

# Add the upstream remote

# Create a new branch for your contribution

# Create a Python virtual environment named 'venv'
echo "Creating a virtual environment named 'venv'..."
python3.10 -m venv venv  # Ensure Python 3.10 or your preferred version is installed

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install PyTorch
echo "Installing PyTorch..."
pip install torch  # Adjust this to match your requirements (e.g., a specific version or with GPU support)

# Install Kornia and development dependencies
echo "Installing Kornia and development dependencies..."
pip install -e .[dev]

# Verify Kornia installation
echo "Verifying Kornia installation..."
python -c "import kornia; print(kornia.__version__)"

echo "Kornia development setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
