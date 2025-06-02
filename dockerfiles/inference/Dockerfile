FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/requirements-dl.txt

# Install requirements
RUN pip install --no-cache-dir -r requirements-dl.txt

# Clone the repository
RUN git clone https://github.com/roboflow/inference.git

# Install the package from source in development mode
WORKDIR /app/inference
RUN pip install -e .

# Set environment variables for API server
ENV PORT=9001
ENV HOST=0.0.0.0
ENV API_KEY=""
ENV ALLOW_ORIGINS="*"
ENV WORKERS=1
ENV DISABLE_ACTIVE_LEARNING=true
ENV MODE=http
ENV ENABLE_METRICS=false

# Expose the API port
EXPOSE 9001

# Start the API server
CMD ["python", "-m", "uvicorn", "inference.core.interfaces.http.http_api:app", "--host", "0.0.0.0", "--port", "9001", "--workers", "1"]