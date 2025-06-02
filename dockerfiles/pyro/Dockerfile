FROM python:3.9-slim

WORKDIR /app

# Install git and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone Pyro repo
RUN git clone https://github.com/pyro-ppl/pyro.git

# Copy requirements file to leverage Docker cache
COPY requirements-dl.txt /app/requirements-dl.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Install Pyro from source
WORKDIR /app/pyro
RUN pip install -e .

# Set working directory to /app
WORKDIR /app

# Command to run when container starts
CMD ["python", "-c", "import pyro; print(f'Pyro version: {pyro.__version__}')"]