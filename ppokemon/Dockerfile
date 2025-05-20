FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Use an entrypoint script for better signal handling
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

# Handle SIGTERM gracefully
trap 'echo "Received SIGTERM. Shutting down..."; exit 0' SIGTERM

# Start the application
exec python3 main.py
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]