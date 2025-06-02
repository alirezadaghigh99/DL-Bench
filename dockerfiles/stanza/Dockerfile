FROM python:3.10-slim

WORKDIR /app

# Install git and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-dl.txt /app/

# Clone Stanza repository
RUN git clone https://github.com/stanfordnlp/stanza.git

# Install dependencies
RUN pip install --no-cache-dir -r requirements-dl.txt

# Build Stanza from source
WORKDIR /app/stanza
RUN pip install -e .

# Set Python path to include Stanza
ENV PYTHONPATH="${PYTHONPATH}:/app/stanza"

# Default command
CMD ["python", "-c", "import stanza; print('Stanza installation successful!')"]