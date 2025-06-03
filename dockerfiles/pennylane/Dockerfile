  FROM python:3.10-slim

  # Install git and build dependencies
  RUN apt-get update && apt-get install -y \
      build-essential \
      git \
      && rm -rf /var/lib/apt/lists/*

  # Set working directory
  WORKDIR /app

  # Copy requirements file
  COPY requirements-dl.txt .

  # Clone PennyLane repository
  RUN git clone https://github.com/PennyLaneAI/pennylane.git

  # Install dependencies from requirements-dl.txt
  # (Note: We remove the PennyLane line since we're installing from source)
  RUN grep -v "^-e git+https://github.com/PennyLaneAI/pennylane.git" requirements-dl.txt > requirements-filtered.txt \
      && pip install -r requirements-filtered.txt

  # Install PennyLane from source
  WORKDIR /app/pennylane
  RUN pip install -e .

  # Set the entrypoint to Python
  ENTRYPOINT ["python"]
