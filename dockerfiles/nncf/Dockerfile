FROM python:3.10-slim

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone NNCF repository
RUN git clone https://github.com/openvinotoolkit/nncf.git /app/nncf

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build NNCF from source
WORKDIR /app/nncf
RUN pip install --no-cache-dir .

# Set PYTHONPATH for running examples
ENV PYTHONPATH="/app/nncf:${PYTHONPATH}"

# Set working directory for running commands
WORKDIR /app/nncf

CMD ["python", "-c", "import nncf; print(f'NNCF version: {nncf.__version__}')"]