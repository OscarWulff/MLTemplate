FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src src/

# Copy data files
COPY data/processed data/processed

# Copy configuration and dependency files
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Create necessary directories
RUN mkdir -p models reports/figures

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/src

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Set the entry point to run the training script
ENTRYPOINT ["python", "-u", "src/machinelearningtemplate/train.py"]
