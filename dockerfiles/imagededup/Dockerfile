FROM python:3.10-slim

# Install git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements-dl.txt .

# Install Cython first (required for building from source)
RUN pip install "cython>=0.29"

# Install dependencies
RUN pip install -r requirements-dl.txt

# Clone the repository
RUN git clone https://github.com/idealo/imagededup.git

# Build and install from source
WORKDIR /app/imagededup
RUN python setup.py install

# Set the working directory back to /app
WORKDIR /app

# Run a simple test to verify installation
RUN python -c "from imagededup.methods import PHash; print('ImageDeDup successfully installed!')"

# Default command
CMD ["python", "-c", "import imagededup; print(f'ImageDeDup version: {imagededup.__version__}')"]