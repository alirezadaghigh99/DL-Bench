FROM python:3.11-slim

# Install git and build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libpng-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/

# Clone torchvision repository
RUN git clone https://github.com/pytorch/vision.git /app/vision

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Set working directory to the cloned repository
WORKDIR /app/vision

# Build torchvision from source (without installing it)
RUN python setup.py build

# Create mount point for code that can be changed
VOLUME /app/vision/torchvision

# Set environment variable to enable GPU support if available
ENV FORCE_CUDA=1

# Default command to run tests
CMD ["pytest", "test", "-vvv"]
