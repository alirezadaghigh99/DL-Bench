FROM python:3.11-slim

# Install git and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt .

# Clone Emukit repository
RUN git clone https://github.com/EmuKit/emukit.git

# Set up Python environment and install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build and install Emukit from source
WORKDIR /app/emukit
RUN pip install -e .

# Run tests to verify installation
RUN pip install -r requirements/test_requirements.txt && \
    pytest tests

# Set the entrypoint to python
WORKDIR /app
ENTRYPOINT ["python"]