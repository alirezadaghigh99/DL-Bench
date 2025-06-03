FROM python:3.9-slim

# Install git and build dependencies
RUN apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Clone the repository (skip this step if you're adding this file to the repo)
RUN git clone https://github.com/itdxer/neupy.git && \
    cd neupy

# Install neupy from source
WORKDIR /app/neupy
RUN pip install -e .

# Set working directory to neupy
WORKDIR /app/neupy

# Default command
CMD ["python", "-c", "import neupy; print(f'NeuPy version: {neupy.__version__}')"]