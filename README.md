# 🎵 Multilingual Audio Intelligence System

![Multilingual Audio Intelligence System Banner](/static/imgs/banner.png)

## Overview

The Multilingual Audio Intelligence System is an advanced AI-powered platform that combines state-of-the-art speaker diarization, automatic speech recognition, and neural machine translation to deliver comprehensive audio analysis capabilities. This sophisticated system processes multilingual audio content, identifies individual speakers, transcribes speech with high accuracy, and provides intelligent translations across multiple languages, transforming raw audio into structured, actionable insights.

## Features

### Demo Mode with Professional Audio Files
- **Yuri Kizaki - Japanese Audio**: Professional voice message about website communication (23 seconds)
- **French Film Podcast**: Discussion about movies including Social Network and Paranormal Activity (25 seconds)
- Smart demo file management with automatic download and preprocessing
- Instant results with cached processing for blazing-fast demonstration

### Enhanced User Interface
- **Audio Waveform Visualization**: Real-time waveform display with HTML5 Canvas
- **Interactive Demo Selection**: Beautiful cards for selecting demo audio files
- **Improved Transcript Display**: Color-coded confidence levels and clear translation sections
- **Professional Audio Preview**: Audio player with waveform visualization

### Screenshots

#### 🎬 Demo Banner

![Demo Banner](/static/imgs/demo_banner.png)

#### 📝 Transcript with Translation

![Transcript with Translation](/static/imgs/demo_res_transcript_translate.png)

#### 📊 Visual Representation

<p align="center">
  <img src="static/imgs/demo_res_visual.png" alt="Visual Output"/>
</p>

#### 🧠 Summary Output

![Summary Output](/static/imgs/demo_res_summary.png)

## Demo & Documentation

- 🎥 [Video Preview]()
- 📄 [Project Documentation](DOCUMENTATION.md)

## Installation and Quick Start

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Prathameshv07/Multilingual-Audio-Intelligence-System.git
   cd Multilingual-Audio-Intelligence-System
   ```

2. **Create and Activate Conda Environment:**
   ```bash
   conda create --name audio_challenge python=3.9
   conda activate audio_challenge
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   ```bash
   cp config.example.env .env
   # Edit .env file with your HUGGINGFACE_TOKEN for accessing gated models
   ```

5. **Preload AI Models (Recommended):**
   ```bash
   python model_preloader.py
   ```

6. **Initialize Application:**
   ```bash
   python run_fastapi.py
   ```

## File Structure

```
Multilingual-Audio-Intelligence-System/
├── web_app.py                      # FastAPI application with RESTful endpoints
├── model_preloader.py              # Intelligent model loading with progress tracking
├── run_fastapi.py                  # Application startup script with preloading
├── src/
│   ├── main.py                     # AudioIntelligencePipeline orchestrator
│   ├── audio_processor.py          # Advanced audio preprocessing and normalization
│   ├── speaker_diarizer.py         # pyannote.audio integration for speaker identification
│   ├── speech_recognizer.py        # faster-whisper ASR with language detection
│   ├── translator.py               # Neural machine translation with multiple models
│   ├── output_formatter.py         # Multi-format result generation and export
│   └── utils.py                    # Utility functions and performance monitoring
├── templates/
│   └── index.html                  # Responsive web interface with home page
├── static/                         # Static assets and client-side resources
├── model_cache/                    # Intelligent model caching directory
├── uploads/                        # User audio file storage
├── outputs/                        # Generated results and downloads
├── requirements.txt                # Comprehensive dependency specification
├── Dockerfile                      # Production-ready containerization
└── config.example.env              # Environment configuration template
```

## Configuration

### Environment Variables
Create a `.env` file:
```env
HUGGINGFACE_TOKEN=hf_your_token_here  # Optional, for gated models
```

### Model Configuration
- **Whisper Model**: tiny/small/medium/large
- **Target Language**: en/es/fr/de/it/pt/zh/ja/ko/ar
- **Device**: auto/cpu/cuda

## Supported Audio Formats

- WAV (recommended)
- MP3
- OGG
- FLAC
- M4A

**Maximum file size**: 100MB  
**Recommended duration**: Under 30 minutes

## Development

### Local Development
```bash
python run_fastapi.py
```

### Production Deployment
```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

## Performance

- **Processing Speed**: 2-14x real-time (depending on model size)
- **Memory Usage**: Optimized with INT8 quantization
- **CPU Optimized**: Works without GPU
- **Concurrent Processing**: Async/await support

## Troubleshooting

### Common Issues

1. **Dependencies**: Use `requirements.txt` for clean installation
2. **Memory**: Use smaller models (tiny/small) for limited hardware
3. **Audio Format**: Convert to WAV if other formats fail
4. **Port Conflicts**: Change port in `run_fastapi.py` if 8000 is occupied

### Error Resolution
- Check logs in terminal output
- Verify audio file format and size
- Ensure all dependencies are installed
- Check available system memory

## Support

- **Documentation**: Check `/api/docs` endpoint
- **System Info**: Use the info button in the web interface
- **Logs**: Monitor terminal output for detailed information 