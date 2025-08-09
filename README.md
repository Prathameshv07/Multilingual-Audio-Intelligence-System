# 🎵 Multilingual Audio Intelligence System

## New Features ✨

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

### Technical Improvements
- Automatic demo file download from original sources
- Cached preprocessing results for instant demo response
- Enhanced error handling for missing or corrupted demo files
- Web Audio API integration for dynamic waveform generation

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application (includes demo file setup)
python run_fastapi.py

# Access the application
# http://127.0.0.1:8000
```

## Demo Mode Usage

1. **Select Demo Mode**: Click the "Demo Mode" button in the header
2. **Choose Audio File**: Select either Japanese or French demo audio
3. **Preview**: Listen to the audio and view the waveform
4. **Process**: Click "Process Audio" for instant results
5. **Explore**: View transcripts, translations, and analytics

## Full Processing Mode

1. **Upload Audio**: Drag & drop or click to upload your audio file
2. **Preview**: View waveform and listen to your audio
3. **Configure**: Select model size and target language
4. **Process**: Real-time processing with progress tracking
5. **Download**: Export results in JSON, SRT, or TXT format

## Features

## System Architecture

### Core Components

- **FastAPI Backend** - Production-ready web framework
- **HTML/TailwindCSS Frontend** - Clean, professional interface
- **Audio Processing Pipeline** - Integrated ML models
- **RESTful API** - Standardized endpoints

### Key Features

- **Speaker Diarization** - Identify "who spoke when"
- **Speech Recognition** - Convert speech to text
- **Language Detection** - Automatic language identification
- **Neural Translation** - Multi-language translation
- **Interactive Visualization** - Waveform analysis
- **Multiple Export Formats** - JSON, SRT, TXT

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **PyTorch** - Deep learning framework
- **pyannote.audio** - Speaker diarization
- **faster-whisper** - Speech recognition
- **Helsinki-NLP** - Neural translation

### Frontend
- **HTML5/CSS3** - Clean markup
- **TailwindCSS** - Utility-first styling
- **JavaScript (Vanilla)** - Client-side logic
- **Plotly.js** - Interactive visualizations
- **Font Awesome** - Professional icons

## API Endpoints

### Core Endpoints
- `GET /` - Main application interface
- `POST /api/upload` - Upload and process audio
- `GET /api/status/{task_id}` - Check processing status
- `GET /api/results/{task_id}` - Retrieve results
- `GET /api/download/{task_id}/{format}` - Download outputs

### Demo Endpoints
- `POST /api/demo-process` - Quick demo processing
- `GET /api/system-info` - System information

## File Structure

```
audio_challenge/
├── web_app.py              # FastAPI application
├── run_fastapi.py          # Startup script
├── requirements.txt  # Dependencies
├── templates/
│   └── index.html          # Main interface
├── src/                    # Core modules
│   ├── main.py            # Pipeline orchestrator
│   ├── audio_processor.py  # Audio preprocessing
│   ├── speaker_diarizer.py # Speaker identification
│   ├── speech_recognizer.py # ASR with language detection
│   ├── translator.py      # Neural machine translation
│   ├── output_formatter.py # Output generation
│   └── utils.py           # Utility functions
├── static/                # Static assets
├── uploads/               # Uploaded files
└── outputs/               # Generated outputs
└── README.md
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

## License

MIT License - See LICENSE file for details

## Support

- **Documentation**: Check `/api/docs` endpoint
- **System Info**: Use the info button in the web interface
- **Logs**: Monitor terminal output for detailed information 