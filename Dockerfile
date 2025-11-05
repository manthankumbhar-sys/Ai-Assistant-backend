# Use a small, secure Python base
FROM python:3.11-slim

# Create working dir
WORKDIR /app

# System deps for building wheels (uvicorn/gunicorn etc. compile cleanly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Cloud Run will set $PORT; default to 8080 for local
ENV PORT=8080
# Ensure Python outputs straight to logs
ENV PYTHONUNBUFFERED=1

# Gunicorn with uvicorn workers: robust for production
# -k uvicorn.workers.UvicornWorker
# 2 workers, 1 thread each; tune later in UI
CMD exec gunicorn features_app:app \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --threads 1 \
    --timeout 120
