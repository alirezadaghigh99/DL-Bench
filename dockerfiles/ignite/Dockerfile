FROM python:3.11-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/requirements-dl.txt

# Clone ignite repository
RUN git clone https://github.com/pytorch/ignite.git

# Install dependencies from requirements-dl.txt
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build ignite from source
WORKDIR /app/ignite
RUN pip install --no-cache-dir -e .

# Set working directory back to /app
WORKDIR /app

# Default command
CMD ["python", "-c", "import ignite; print(f'PyTorch Ignite version: {ignite.__version__}')"]