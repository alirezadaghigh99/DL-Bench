FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt .

# Remove the line that references the git repository
RUN sed -i '/^-e git/d' requirements-dl.txt

# Install Python dependencies
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements-dl.txt

# Clone DeepChem repository
RUN git clone https://github.com/deepchem/deepchem.git

# Build and install DeepChem from source
WORKDIR /app/deepchem
RUN pip install --no-cache-dir -e .

# Set the working directory back to /app
WORKDIR /app

# Default command
CMD ["python"]