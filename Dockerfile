FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — cached layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY . .

# Create runtime dirs
RUN mkdir -p logs models_saved data

# Expose API port
EXPOSE 8000

# Default: start API (override with docker run ... python scripts/train_models.py)
CMD ["python", "scripts/start_api.py"]
