FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt .

# Clone scikit-learn repository
RUN git clone https://github.com/scikit-learn/scikit-learn.git

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build scikit-learn from source
WORKDIR /app/scikit-learn
RUN pip install --no-cache-dir --verbose --no-build-isolation --editable . \
    --config-settings editable-verbose=true

# Verify installation
RUN python -c "import sklearn; sklearn.show_versions()"

# Set the default command
CMD ["python", "-c", "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"]