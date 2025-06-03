FROM python:3.9-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone Lightly repository
RUN git clone https://github.com/lightly-ai/lightly.git /app/lightly

# Copy requirements-dl.txt (this assumes you'll place it alongside the Dockerfile)
COPY requirements-dl.txt /app/

# Install requirements
RUN pip install --no-cache-dir -r /app/requirements-dl.txt

# Install Lightly from source (using the local requirements already installed)
WORKDIR /app/lightly
RUN pip install --no-cache-dir -e .

# Set default command
CMD ["python", "-c", "import lightly; print(f'Lightly version: {lightly.__version__}')"]