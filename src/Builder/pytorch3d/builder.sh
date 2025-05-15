#!/bin/bash

# Script to install PyTorch3D from source and its dependencies on Ubuntu

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install gcc and g++
echo "Installing gcc and g++..."
sudo apt-get install -y gcc g++

# Install Anaconda or Miniconda if not already installed (optional)
echo "Checking for Conda installation..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init
    source ~/.bashrc
else
    echo "Conda is already installed."
fi

# Create and activate a new Conda environment
echo "Creating and activating a Conda environment named 'pytorch3d'..."
conda create -n pytorch3d python=3.9 -y
conda activate pytorch3d

# Install PyTorch and torchvision with CUDA support
echo "Installing PyTorch and torchvision with CUDA support..."
conda install pytorch=2.1.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y

# Install ioPath
echo "Installing ioPath..."
conda install -c iopath iopath -y

# Install CUB if using CUDA version older than 11.7
echo "Checking CUDA version..."
CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
if [[ $(echo "$CUDA_VERSION < 11.7" | bc -l) -eq 1 ]]; then
    echo "Installing CUB library..."
    conda install -c bottler nvidiacub -y
fi

# Install additional packages for demos, linting, and tests
echo "Installing packages for demos and tests..."
conda install jupyter -y
pip install scikit-image matplotlib imageio plotly opencv-python

echo "Installing linting and testing tools..."
conda install -c fvcore -c conda-forge fvcore -y
pip install black usort flake8 flake8-bugbear flake8-comprehensions

# Clone PyTorch3D repository and install from source
echo "Cloning PyTorch3D repository..."
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d

echo "Installing PyTorch3D from source..."
pip install -e .

# Optional: Rebuild PyTorch3D if necessary
# echo "Rebuilding PyTorch3D..."
# rm -rf build/ **/*.so
# pip install -e .

echo "Installation complete! PyTorch3D has been installed from source along with its dependencies."
