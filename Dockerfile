# Small, secure base
FROM python:3.11-slim

# Workdir inside the container
WORKDIR /app

# System deps for wheels (fastapi/uvicorn often fine, but this keeps builds smooth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache friendly)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . /app

# Cloud Run will inject $PORT; default to 8080 for local runs
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start FastAPI with uvicorn (2 workers is a good default)
# IMPORTANT: "features_app:app" must match <python_file_without_py>:<fastapi_app_variable>
CMD exec uvicorn features_app:app --host 0.0.0.0 --port $PORT --workers 2
