FROM python:3.7-slim

# Install git and build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Clone PFRL repository
RUN git clone https://github.com/pfnet/pfrl.git

# Install PFRL from source
WORKDIR /app/pfrl
RUN python setup.py install

# Set PYTHONPATH to include PFRL
ENV PYTHONPATH="/app/pfrl:${PYTHONPATH}"

# Set the default command
CMD ["python", "-c", "import pfrl; print(f'PFRL version: {pfrl.__version__}')"]