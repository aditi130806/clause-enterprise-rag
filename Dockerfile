# Base Python Slim Image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for C-extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY . .

# Expose default Cloud Run port
EXPOSE 8080

# Environment defaults for Cloud Run container execution
ENV PORT=8080 \
    APP_ENV=production

# Healthcheck for container readiness
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Launch Streamlit server bound to 0.0.0.0 and dynamic $PORT
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true"]
