FROM python:3.9-slim

WORKDIR /app

# Install git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-dl.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Clone Laplace repository
RUN git clone https://github.com/aleximmer/Laplace.git

# Install Laplace in development mode
WORKDIR /app/Laplace
RUN pip install -e ".[dev]"

# Run tests to verify installation
RUN python -m pytest

# Set the default command
CMD ["python", "-c", "import laplace; print('Laplace installation successful!')"]