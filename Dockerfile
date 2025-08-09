FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p templates static uploads outputs model_cache

# Set environment variables for HuggingFace Spaces
ENV PYTHONPATH=/app
ENV GRADIO_ANALYTICS_ENABLED=False

# Preload models during build time (optional - comment out if build time is too long)
# RUN python model_preloader.py

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/system-info || exit 1

# Start command for HuggingFace Spaces
CMD ["python", "-c", "import subprocess; subprocess.run(['python', 'model_preloader.py']); import uvicorn; uvicorn.run('web_app:app', host='0.0.0.0', port=7860, workers=1)"] 