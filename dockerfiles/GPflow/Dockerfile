FROM python:3.10-slim

# Install git and build essentials
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone GPflow repository
RUN git clone https://github.com/GPflow/GPflow.git

# Copy requirements file
COPY requirements-dl.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build GPflow from source
WORKDIR /app/GPflow
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/GPflow:$PYTHONPATH

# Default command
CMD ["python", "-c", "import gpflow; print(f'GPflow version: {gpflow.__version__}')"]