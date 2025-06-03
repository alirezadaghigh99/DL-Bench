# syntax=docker/dockerfile:1

# Base image with essential build tools
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    openjdk-11-jdk \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/requirements-dl.txt

# Set environment variables for Java and Python
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${PATH}:${JAVA_HOME}/bin"
ENV PYTHONPATH=/app/recommenders

# Clone recommenders repository
RUN git clone https://github.com/recommenders-team/recommenders.git /app/recommenders

# Create and activate Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:${PATH}"

# Install pip requirements first
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-dl.txt

# Install recommenders package from source
WORKDIR /app/recommenders
RUN pip install -e .

# Create Jupyter kernel
RUN python -m ipykernel install --user --name recommenders --display-name "Recommenders"

# Expose Jupyter port
EXPOSE 8888

# Default command to start Jupyter notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]