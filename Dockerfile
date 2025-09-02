FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p templates static uploads outputs model_cache temp_files demo_results demo_audio \
    && chmod -R 755 templates static uploads outputs model_cache temp_files demo_results demo_audio

# Set environment variables for Hugging Face Spaces
ENV PYTHONPATH=/app \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_MODELS_CACHE=/app/model_cache \
    OUTPUT_DIR=./outputs \
    TEMP_DIR=./temp_files \
    WHISPER_MODEL_SIZE=small \
    TARGET_LANGUAGE=en \
    MAX_WORKERS=1 \
    USE_GPU=false \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    TORCH_HOME=/app/model_cache \
    XDG_CACHE_HOME=/app/model_cache \
    PYANNOTE_CACHE=/app/model_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HUGGINGFACE_HUB_CACHE=/app/model_cache \
    HF_HUB_CACHE=/app/model_cache

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Health check for Hugging Face Spaces
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/system-info || exit 1

# Preload models and start the application
CMD ["python", "-c", "import subprocess; import time; print('🚀 Starting Enhanced Multilingual Audio Intelligence System...'); subprocess.run(['python', 'model_preloader.py']); print('✅ Models loaded successfully'); import uvicorn; uvicorn.run('web_app:app', host='0.0.0.0', port=7860, workers=1, log_level='info')"]