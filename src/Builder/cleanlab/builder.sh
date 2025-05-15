#!/bin/bash

# Exit the script if any command fails
set -e

# Clone the cleanlab repository
echo "Cloning the cleanlab repository..."
git clone https://github.com/cleanlab/cleanlab.git
cd cleanlab

# Set up a virtual environment
echo "Creating a virtual environment..."
python3 -m venv ./venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source ./venv/bin/activate

# Install dependencies
echo "Installing development requirements..."
pip install -r requirements-dev.txt

# Install cleanlab as an editable package
echo "Installing cleanlab as an editable package..."
pip install -e .

# If running on a Mac with Apple silicon, replace TensorFlow packages
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Detected Apple silicon. Replacing TensorFlow packages..."
  sed -i '' 's/^tensorflow.*/tensorflow-macos==2.9.2/' requirements-dev.txt
  echo 'tensorflow-metal==0.5.1' >> requirements-dev.txt
  pip install -r requirements-dev.txt
fi

echo "Setup complete. To activate the virtual environment in future sessions, run:"
echo "source ./venv/bin/activate"
