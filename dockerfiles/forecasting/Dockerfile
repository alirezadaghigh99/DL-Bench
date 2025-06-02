FROM python:3.11-slim

# Install git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-dl.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Clone the pytorch-forecasting repository
RUN git clone https://github.com/sktime/pytorch-forecasting.git && \
    cd pytorch-forecasting && \
    git checkout e08f4e6d23737c36949d704215ccbbbaa84fc1b6

# Install pytorch-forecasting from source
WORKDIR /app/pytorch-forecasting
RUN pip install --no-cache-dir -e .

# Set the Python path to include the pytorch-forecasting package
ENV PYTHONPATH="${PYTHONPATH}:/app/pytorch-forecasting"

# Set the default command to python
CMD ["python"]