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
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libsndfile1-dev \
    libflac-dev \
    libvorbis-dev \
    libogg-dev \
    libmp3lame-dev \
    libmad0-dev \
    libtwolame-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with proper error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p templates static uploads outputs model_cache temp_files demo_results demo_audio \
    /tmp/matplotlib /tmp/fontconfig \
    && chmod -R 777 templates static \
    && chmod -R 777 uploads outputs model_cache temp_files demo_results demo_audio \
    && chmod -R 777 /tmp/matplotlib /tmp/fontconfig

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
    HF_HUB_CACHE=/app/model_cache \
    FONTCONFIG_PATH=/tmp/fontconfig \
    # Fix for audio processing libraries
    CTRANSLATE2_FORCE_CPU_ISA=generic \
    # Disable problematic features
    TF_CPP_MIN_LOG_LEVEL=2 \
    TOKENIZERS_PARALLELISM=false \
    # Fix executable stack issues
    ONNX_EXECUTION_PROVIDER=cpu \
    # Disable problematic optimizations
    OMP_NUM_THREADS=1

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Health check for Hugging Face Spaces
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Preload models and start the application
CMD ["python", "startup.py"]