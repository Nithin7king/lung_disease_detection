FROM python:3.10-slim

# Install system dependencies needed by librosa (soundfile/libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer cache optimization)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full backend application
COPY backend/ ./

# Expose the port (Render injects $PORT at runtime)
EXPOSE 5000

# Run with gunicorn for production (same as the original Procfile)
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} app:app
