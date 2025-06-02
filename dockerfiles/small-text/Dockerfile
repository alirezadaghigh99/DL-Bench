FROM python:3.11-slim

# Install git and build essentials for any compilation requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements-dl.txt
COPY requirements-dl.txt .

# Clone small-text repository
RUN git clone https://github.com/webis-de/small-text.git

# Install requirements
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build and install small-text from source
WORKDIR /app/small-text
RUN pip install --no-cache-dir -e .

# Set Python path to include small-text
ENV PYTHONPATH="${PYTHONPATH}:/app/small-text"

# Default command - can be overridden
CMD ["python", "-c", "import small_text; print(f'Small-Text version: {small_text.__version__}')"]