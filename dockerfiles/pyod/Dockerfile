FROM python:3.11-slim

WORKDIR /app

# Copy requirements-dl.txt (should be available alongside the Dockerfile)
COPY requirements-dl.txt .

# Install git and other build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone PyOD repository
RUN git clone https://github.com/yzhao062/pyod.git

# Install the dependencies from requirements-dl.txt
RUN pip install --no-cache-dir -r requirements-dl.txt

# Install PyOD from source
WORKDIR /app/pyod
RUN pip install .

# Set up a default command (can be overridden at runtime)
CMD ["python", "-c", "import pyod; print(f'PyOD version {pyod.__version__} installed successfully!')"]