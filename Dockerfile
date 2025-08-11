FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories & fix permissions
RUN mkdir -p templates static uploads outputs model_cache temp_files demo_results \
    && chmod -R 777 templates static uploads outputs model_cache temp_files demo_results

# Environment variables
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
    MPLCONFIGDIR=/tmp/matplotlib


EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/system-info || exit 1

CMD ["python", "-c", "import subprocess; subprocess.run(['python', 'model_preloader.py']); import uvicorn; uvicorn.run('web_app:app', host='0.0.0.0', port=7860, workers=1)"]