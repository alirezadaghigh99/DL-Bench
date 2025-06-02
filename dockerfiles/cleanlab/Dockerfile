FROM python:3.10-slim

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/cleanlab/cleanlab.git /app/cleanlab

# Set working directory to the cloned repo
WORKDIR /app/cleanlab

# Copy your local requirements file
COPY requirements-dl.txt .

# Install dependencies from your requirements file
RUN pip install --no-cache-dir -r requirements-dl.txt

# Install the package in development mode
RUN pip install -e .

# Create a volume mount point for code changes
VOLUME ["/app/cleanlab"]

# Default command to run tests
CMD ["pytest", "tests/test_classification.py", "-v"]
