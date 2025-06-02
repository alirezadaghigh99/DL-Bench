  FROM python:3.9-slim

  # Install git and other dependencies
  RUN apt-get update && apt-get install -y \
      git \
      build-essential \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

  # Set working directory
  WORKDIR /avalanche

  # Clone the Avalanche repository
  RUN git clone https://github.com/ContinualAI/avalanche.git .

  # Install PyTorch dependencies
  RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

  # Install Avalanche in development mode with all extras
  RUN pip install -e ".[dev,all]"

  # Verify installation
  RUN python -c "import avalanche; print(avalanche.__version__)"

  # Set the default command
  CMD ["python"]
