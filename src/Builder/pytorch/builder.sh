#!/bin/bash

# Script to set up and install PyTorch from source on Ubuntu

# Function to display messages
function print_message {
    echo "===================================="
    echo "$1"
    echo "===================================="
}

# Update and upgrade system packages
print_message "Updating system packages"
sudo apt-get update && sudo apt-get upgrade -y

# Check if Conda is installed and install Miniconda if not found
print_message "Checking for Conda installation"
if ! command -v conda &> /dev/null; then
    print_message "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init
    source ~/.bashrc
else
    print_message "Conda is already installed"
fi

# Create and activate a new Conda environment
CONDA_ENV_NAME="pytorch"
print_message "Creating and activating Conda environment: $CONDA_ENV_NAME"
conda create -y -n $CONDA_ENV_NAME python=3.9
conda activate $CONDA_ENV_NAME

# Install dependencies
print_message "Installing dependencies"
conda install -y cmake ninja
pip install -r requirements.txt
pip install mkl-static mkl-include

# Install CUDA-specific libraries if CUDA is present
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
    print_message "CUDA detected. Installing MAGMA for CUDA version $CUDA_VERSION"
    conda install -y -c pytorch magma-cuda${CUDA_VERSION//./}
else
    print_message "CUDA not detected. Skipping CUDA-specific installations"
fi

# Clone PyTorch source code
PYTORCH_DIR="pytorch"
print_message "Cloning PyTorch repository"
if [ ! -d "$PYTORCH_DIR" ]; then
    git clone --recursive https://github.com/pytorch/pytorch.git
else
    print_message "PyTorch directory already exists. Skipping clone"
fi

cd pytorch

# Update submodules if necessary
print_message "Updating Git submodules"
git submodule sync
git submodule update --init --recursive

# Set up the environment for building PyTorch
print_message "Setting up environment variables"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
export _GLIBCXX_USE_CXX11_ABI=1  # Enable new C++ ABI if needed

# Compile for ROCm if applicable (uncomment if ROCm support is desired)
# print_message "Compiling for AMD ROCm"
# python tools/amd_build/build_amd.py

# Build and install PyTorch
print_message "Building and installing PyTorch"
python setup.py develop

print_message "PyTorch installation from source completed!"
