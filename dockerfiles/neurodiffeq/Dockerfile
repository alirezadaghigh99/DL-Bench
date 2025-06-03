# Use Python 3.8 slim as base image
FROM python:3.8-slim-buster

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-dl.txt

# Clone the neurodiffeq repository
# Note: We're not using the line from requirements-dl.txt because we want to build from source
RUN git clone https://github.com/NeuroDiffGym/neurodiffeq.git /app/neurodiffeq

# Install neurodiffeq from source
WORKDIR /app/neurodiffeq
RUN pip install -e .

# Run tests to verify installation


# Set default command to python
CMD ["python"]
