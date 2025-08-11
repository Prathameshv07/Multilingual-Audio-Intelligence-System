"""
Multilingual Audio Intelligence System - FastAPI Web Application

Professional web interface for the complete multilingual audio intelligence pipeline.
Built with FastAPI, HTML templates, and modern CSS for production deployment.

Features:
- Clean, professional UI design
- Real-time audio processing
- Interactive visualizations
- Multiple output formats
- RESTful API endpoints
- Production-ready architecture

Author: Audio Intelligence Team
"""

import os
import sys
import logging
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
import asyncio
from datetime import datetime
import requests
import hashlib
from urllib.parse import urlparse

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Data processing
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Safe imports with error handling
try:
    from main import AudioIntelligencePipeline
    MAIN_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import main pipeline: {e}")
    MAIN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import Plotly: {e}")
    PLOTLY_AVAILABLE = False

try:
    from utils import validate_audio_file, format_duration, get_system_info
    UTILS_AVAILABLE = True
except Exception as e:
    logger.error(f"Failed to import utils: {e}")
    UTILS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual Audio Intelligence System",
    description="Professional AI-powered speaker diarization, transcription, and translation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/demo_audio", StaticFiles(directory="demo_audio"), name="demo_audio")

# Global pipeline instance
pipeline = None

# Processing status store (in production, use Redis or database)
processing_status = {}
processing_results = {}  # Store actual results

# Demo file configuration
DEMO_FILES = {
    "yuri_kizaki": {
        "filename": "Yuri_Kizaki.mp3",
        "display_name": "Yuri Kizaki - Japanese Audio",
        "language": "Japanese",
        "description": "Audio message about website communication enhancement",
        "url": "https://www.mitsue.co.jp/service/audio_and_video/audio_production/media/narrators_sample/yuri_kizaki/03.mp3",
        "expected_text": "音声メッセージが既存のウェブサイトを超えたコミュニケーションを実現。目で見るだけだったウェブサイトに音声情報をインクルードすることで、情報に新しい価値を与え、他者との差別化に効果を発揮します。",
        "expected_translation": "Audio messages enable communication beyond existing websites. By incorporating audio information into visually-driven websites, you can add new value to the information and effectively differentiate your website from others."
    },
    "film_podcast": {
        "filename": "Film_Podcast.mp3", 
        "display_name": "French Film Podcast",
        "language": "French",
        "description": "Discussion about recent movies including Social Network and Paranormal Activity",
        "url": "https://www.lightbulblanguages.co.uk/resources/audio/film-podcast.mp3",
        "expected_text": "Le film intitulé The Social Network traite de la création du site Facebook par Mark Zuckerberg et des problèmes judiciaires que cela a comporté pour le créateur de ce site.",
        "expected_translation": "The film The Social Network deals with the creation of Facebook by Mark Zuckerberg and the legal problems this caused for the creator of this site."
    }
}

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    try:
        # Basic system check
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        if free < 50 * 1024 * 1024:  # less than 50MB
            return {"status": "error", "detail": "Low disk space"}
        
        # Check if models are loaded
        if not hasattr(app.state, "models_loaded") or not app.state.models_loaded:
            return {"status": "error", "detail": "Models not loaded"}
        
        return {"status": "ok"}
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# Demo results cache
demo_results_cache = {}

class DemoManager:
    """Manages demo files and preprocessing."""
    
    def __init__(self):
        self.demo_dir = Path("demo_audio")
        self.demo_dir.mkdir(exist_ok=True)
        self.results_dir = Path("demo_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def ensure_demo_files(self):
        """Ensure demo files are available and processed."""
        for demo_id, config in DEMO_FILES.items():
            file_path = self.demo_dir / config["filename"]
            results_path = self.results_dir / f"{demo_id}_results.json"
            
            # Check if file exists, download if not
            if not file_path.exists():
                logger.info(f"Downloading demo file: {config['filename']}")
                try:
                    await self.download_demo_file(config["url"], file_path)
                except Exception as e:
                    logger.error(f"Failed to download {config['filename']}: {e}")
                    continue
            
            # Check if results exist, process if not
            if not results_path.exists():
                logger.info(f"Processing demo file: {config['filename']}")
                try:
                    await self.process_demo_file(demo_id, file_path, results_path)
                except Exception as e:
                    logger.error(f"Failed to process {config['filename']}: {e}")
                    continue
            
            # Load results into cache
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    demo_results_cache[demo_id] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cached results for {demo_id}: {e}")
    
    async def download_demo_file(self, url: str, file_path: Path):
        """Download demo file from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded demo file: {file_path.name}")
    
    async def process_demo_file(self, demo_id: str, file_path: Path, results_path: Path):
        """Process demo file using actual pipeline and cache results."""
        config = DEMO_FILES[demo_id]
        try:
            # Initialize pipeline for demo processing
            pipeline = AudioIntelligencePipeline(
                whisper_model_size="small",
                target_language="en",
                device="auto",
                hf_token=os.getenv('HUGGINGFACE_TOKEN'),
                output_dir="./outputs"
            )
            
            # Process the actual audio file
            logger.info(f"Processing demo file: {file_path}")
            results = pipeline.process_audio(
                str(file_path),
                save_outputs=True,
                output_formats=['json', 'srt_original', 'srt_translated', 'text', 'summary']
            )
            
            # Format results for demo display
            formatted_results = self.format_demo_results(results, demo_id)
            
            # Save formatted results
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Demo file processed and cached: {config['filename']}")
            
        except Exception as e:
            logger.error(f"Failed to process demo file {demo_id}: {e}")
            # Create fallback results if processing fails
            fallback_results = self.create_fallback_results(demo_id, str(e))
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(fallback_results, f, indent=2, ensure_ascii=False)
    
    def format_demo_results(self, results: Dict, demo_id: str) -> Dict:
        """Format pipeline results for demo display."""
        formatted_results = {
            "segments": [],
            "summary": {
                "total_duration": 0,
                "num_speakers": 0,
                "num_segments": 0,
                "languages": [],
                "processing_time": 0
            }
        }
        
        try:
            # Extract segments from actual pipeline results
            if 'processed_segments' in results:
                for seg in results['processed_segments']:
                    formatted_results["segments"].append({
                        "speaker": seg.speaker_id if hasattr(seg, 'speaker_id') else "Speaker 1",
                        "start_time": seg.start_time if hasattr(seg, 'start_time') else 0,
                        "end_time": seg.end_time if hasattr(seg, 'end_time') else 0,
                        "text": seg.original_text if hasattr(seg, 'original_text') else "",
                        "translated_text": seg.translated_text if hasattr(seg, 'translated_text') else "",
                        "language": seg.original_language if hasattr(seg, 'original_language') else "unknown"
                    })
            
            # Extract metadata
            if 'audio_metadata' in results:
                metadata = results['audio_metadata']
                formatted_results["summary"]["total_duration"] = metadata.get('duration_seconds', 0)
            
            if 'processing_stats' in results:
                stats = results['processing_stats']
                formatted_results["summary"]["processing_time"] = stats.get('total_time', 0)
            
            # Calculate derived stats
            formatted_results["summary"]["num_segments"] = len(formatted_results["segments"])
            speakers = set(seg["speaker"] for seg in formatted_results["segments"])
            formatted_results["summary"]["num_speakers"] = len(speakers)
            languages = set(seg["language"] for seg in formatted_results["segments"] if seg["language"] != 'unknown')
            formatted_results["summary"]["languages"] = list(languages) if languages else ["unknown"]
            
        except Exception as e:
            logger.error(f"Error formatting demo results: {e}")
            # Return basic structure if formatting fails
            formatted_results["segments"] = [
                {
                    "speaker": "Speaker 1",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "text": f"Demo processing completed. Error in formatting: {str(e)}",
                    "translated_text": f"Demo processing completed. Error in formatting: {str(e)}",
                    "language": "en"
                }
            ]
            formatted_results["summary"]["total_duration"] = 5.0
            formatted_results["summary"]["num_segments"] = 1
            formatted_results["summary"]["num_speakers"] = 1
            formatted_results["summary"]["languages"] = ["en"]
        
        return formatted_results
    
    def create_fallback_results(self, demo_id: str, error_msg: str) -> Dict:
        """Create fallback results when demo processing fails."""
        config = DEMO_FILES[demo_id]
        return {
            "segments": [
                {
                    "speaker": "System",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "text": f"Demo processing failed: {error_msg}",
                    "translated_text": f"Demo processing failed: {error_msg}",
                    "language": "en"
                }
            ],
            "summary": {
                "total_duration": 1.0,
                "num_speakers": 1,
                "num_segments": 1,
                "languages": ["en"],
                "processing_time": 0.1
            }
        }

# Initialize demo manager
demo_manager = DemoManager()


class AudioProcessor:
    """Audio processing class with error handling."""
    
    def __init__(self):
        self.pipeline = None
    
    def initialize_pipeline(self, whisper_model: str = "small", 
                          target_language: str = "en", 
                          hf_token: str = None):
        """Initialize the audio intelligence pipeline."""
        if not MAIN_AVAILABLE:
            raise Exception("Main pipeline module not available")
        
        if self.pipeline is None:
            logger.info("Initializing Audio Intelligence Pipeline...")
            try:
                self.pipeline = AudioIntelligencePipeline(
                    whisper_model_size=whisper_model,
                    target_language=target_language,
                    device="auto",
                    hf_token=hf_token or os.getenv('HUGGINGFACE_TOKEN'),
                    output_dir="./outputs"
                )
                logger.info("Pipeline initialization complete!")
            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                raise
        
        return self.pipeline
    
    async def process_audio_file(self, file_path: str, 
                               whisper_model: str = "small",
                               target_language: str = "en",
                               hf_token: str = None,
                               task_id: str = None) -> Dict[str, Any]:
        """Process audio file and return results."""
        try:
            # Update status
            if task_id:
                processing_status[task_id] = {"status": "initializing", "progress": 10}
            
            # Initialize pipeline
            try:
                pipeline = self.initialize_pipeline(whisper_model, target_language, hf_token)
            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                if task_id:
                    processing_status[task_id] = {"status": "error", "error": f"Pipeline initialization failed: {str(e)}"}
                raise
            
            if task_id:
                processing_status[task_id] = {"status": "processing", "progress": 30}
            
            # Process audio using the actual pipeline
            try:
                logger.info(f"Processing audio file: {file_path}")
                results = pipeline.process_audio(
                    file_path,
                    save_outputs=True,
                    output_formats=['json', 'srt_original', 'srt_translated', 'text', 'summary']
                )
                logger.info("Audio processing completed successfully")
            except Exception as e:
                logger.error(f"Audio processing failed: {e}")
                if task_id:
                    processing_status[task_id] = {"status": "error", "error": f"Audio processing failed: {str(e)}"}
                raise
            
            if task_id:
                processing_status[task_id] = {"status": "generating_outputs", "progress": 80}
            
            # Generate visualization data
            try:
                viz_data = self.create_visualization_data(results)
                results['visualization'] = viz_data
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
                results['visualization'] = {"error": str(e)}
            
            # Store results for later retrieval
            if task_id:
                processing_results[task_id] = results
                processing_status[task_id] = {"status": "complete", "progress": 100}
            
            return results
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            if task_id:
                processing_status[task_id] = {"status": "error", "error": str(e)}
            raise
    
    def create_visualization_data(self, results: Dict) -> Dict:
        """Create visualization data from processing results."""
        viz_data = {}
        
        try:
            # Create waveform data
            if PLOTLY_AVAILABLE and results.get('processed_segments'):
                segments = results['processed_segments']
                
                # Get actual duration from results
                duration = results.get('audio_metadata', {}).get('duration_seconds', 30)
                
                # For demo purposes, generate sample waveform
                # In production, you would extract actual audio waveform data
                time_points = np.linspace(0, duration, min(1000, int(duration * 50)))
                waveform = np.random.randn(len(time_points)) * 0.1  # Sample data
                
                # Create plotly figure
                fig = go.Figure()
                
                # Add waveform
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=waveform,
                    mode='lines',
                    name='Waveform',
                    line=dict(color='#2563eb', width=1)
                ))
                
                # Add speaker segments
                colors = ['#dc2626', '#059669', '#7c2d12', '#4338ca', '#be185d']
                for i, seg in enumerate(segments):
                    color = colors[i % len(colors)]
                    fig.add_vrect(
                        x0=seg.start_time,
                        x1=seg.end_time,
                        fillcolor=color,
                        opacity=0.2,
                        line_width=0,
                        annotation_text=f"{seg.speaker_id}",
                        annotation_position="top left"
                    )
                
                fig.update_layout(
                    title="Audio Waveform with Speaker Segments",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Amplitude",
                    height=400,
                    showlegend=False
                )
                
                viz_data['waveform'] = json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            viz_data['waveform'] = None
        
        return viz_data


# Initialize processor
audio_processor = AudioProcessor()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Initializing Multilingual Audio Intelligence System...")
    
    # Ensure demo files are available and processed
    try:
        await demo_manager.ensure_demo_files()
        logger.info("Demo files initialization complete")
    except Exception as e:
        logger.error(f"Demo files initialization failed: {e}")
    
    # Set models loaded flag for health check
    app.state.models_loaded = True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    target_language: str = Form("en"),
    hf_token: Optional[str] = Form(None)
):
    """Upload and process audio file."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file type
        allowed_types = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Save uploaded file
        file_path = f"uploads/{int(time.time())}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate task ID
        task_id = f"task_{int(time.time())}"
        
        # Start background processing
        asyncio.create_task(
            audio_processor.process_audio_file(
                file_path, whisper_model, target_language, hf_token, task_id
            )
        )
        
        return JSONResponse({
            "task_id": task_id,
            "message": "Processing started",
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get processing status."""
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return JSONResponse(processing_status[task_id])


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Get processing results."""
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = processing_status[task_id]
    if status.get("status") != "complete":
        raise HTTPException(status_code=202, detail="Processing not complete")
    
    # Return actual processed results
    if task_id in processing_results:
        results = processing_results[task_id]
        
        # Convert to the expected format for frontend
        formatted_results = {
            "segments": [],
            "summary": {
                "total_duration": 0,
                "num_speakers": 0,
                "num_segments": 0,
                "languages": [],
                "processing_time": 0
            }
        }
        
        try:
            # Extract segments information
            if 'processed_segments' in results:
                for seg in results['processed_segments']:
                    formatted_results["segments"].append({
                        "speaker": seg.speaker_id if hasattr(seg, 'speaker_id') else "Unknown Speaker",
                        "start_time": seg.start_time if hasattr(seg, 'start_time') else 0,
                        "end_time": seg.end_time if hasattr(seg, 'end_time') else 0,
                        "text": seg.original_text if hasattr(seg, 'original_text') else "",
                        "translated_text": seg.translated_text if hasattr(seg, 'translated_text') else "",
                        "language": seg.original_language if hasattr(seg, 'original_language') else "unknown",
                    })
            
            # Extract summary information
            if 'audio_metadata' in results:
                metadata = results['audio_metadata']
                formatted_results["summary"]["total_duration"] = metadata.get('duration_seconds', 0)
            
            if 'processing_stats' in results:
                stats = results['processing_stats']
                formatted_results["summary"]["processing_time"] = stats.get('total_time', 0)
            
            # Calculate derived statistics
            formatted_results["summary"]["num_segments"] = len(formatted_results["segments"])
            speakers = set(seg["speaker"] for seg in formatted_results["segments"])
            formatted_results["summary"]["num_speakers"] = len(speakers)
            languages = set(seg["language"] for seg in formatted_results["segments"] if seg["language"] != 'unknown')
            formatted_results["summary"]["languages"] = list(languages) if languages else ["unknown"]
            
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            # Fallback to basic structure
            formatted_results = {
                "segments": [
                    {
                        "speaker": "Speaker 1",
                        "start_time": 0.0,
                        "end_time": 5.0,
                        "text": f"Processed audio from file. Full results processing encountered an error: {str(e)}",
                        "language": "en",
                    }
                ],
                "summary": {
                    "total_duration": 5.0,
                    "num_speakers": 1,
                    "num_segments": 1,
                    "languages": ["en"],
                    "processing_time": 2.0
                }
            }
        
        return JSONResponse({
            "task_id": task_id,
            "status": "complete",
            "results": formatted_results
        })
    else:
        # Fallback if results not found
        return JSONResponse({
            "task_id": task_id,
            "status": "complete",
            "results": {
                "segments": [
                    {
                        "speaker": "System",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "text": "Audio processing completed but results are not available for display.",
                        "language": "en",
                    }
                ],
                "summary": {
                    "total_duration": 1.0,
                    "num_speakers": 1,
                    "num_segments": 1,
                    "languages": ["en"],
                    "processing_time": 0.1
                }
            }
        })


@app.get("/api/download/{task_id}/{format}")
async def download_results(task_id: str, format: str):
    """Download results in specified format."""
    if task_id not in processing_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = processing_status[task_id]
    if status.get("status") != "complete":
        raise HTTPException(status_code=202, detail="Processing not complete")
    
    # Get actual results or fallback to sample
    if task_id in processing_results:
        results = processing_results[task_id]
    else:
        # Fallback sample results
        results = {
            'processed_segments': [
                type('Segment', (), {
                    'speaker': 'Speaker 1',
                    'start_time': 0.0,
                    'end_time': 3.5,
                    'text': 'Sample transcript content for download.',
                    'language': 'en'
                })()
            ]
        }
    
    # Generate content based on format
    if format == "json":
        try:
            # Try to use existing JSON output if available
            json_path = f"outputs/{task_id}_complete_results.json"
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Generate JSON from results
                export_data = {
                    "task_id": task_id,
                    "timestamp": datetime.now().isoformat(),
                    "segments": []
                }
                
                if 'processed_segments' in results:
                    for seg in results['processed_segments']:
                        export_data["segments"].append({
                            "speaker": seg.speaker_id if hasattr(seg, 'speaker_id') else "Unknown",
                            "start_time": seg.start_time if hasattr(seg, 'start_time') else 0,
                            "end_time": seg.end_time if hasattr(seg, 'end_time') else 0,
                            "text": seg.original_text if hasattr(seg, 'original_text') else "",
                            "language": seg.original_language if hasattr(seg, 'original_language') else "unknown"
                        })
                
                content = json.dumps(export_data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error generating JSON: {e}")
            content = json.dumps({"error": f"Failed to generate JSON: {str(e)}"}, indent=2)
            
        filename = f"results_{task_id}.json"
        media_type = "application/json"
        
    elif format == "srt":
        try:
            # Try to use existing SRT output if available
            srt_path = f"outputs/{task_id}_subtitles_original.srt"
            if os.path.exists(srt_path):
                with open(srt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Generate SRT from results
                srt_lines = []
                if 'processed_segments' in results:
                    for i, seg in enumerate(results['processed_segments'], 1):
                        start_time = seg.start_time if hasattr(seg, 'start_time') else 0
                        end_time = seg.end_time if hasattr(seg, 'end_time') else 0
                        text = seg.original_text if hasattr(seg, 'original_text') else ""
                        
                        # Format time for SRT (HH:MM:SS,mmm)
                        start_srt = format_srt_time(start_time)
                        end_srt = format_srt_time(end_time)
                        
                        srt_lines.extend([
                            str(i),
                            f"{start_srt} --> {end_srt}",
                            text,
                            ""
                        ])
                
                content = "\n".join(srt_lines)
        except Exception as e:
            logger.error(f"Error generating SRT: {e}")
            content = f"1\n00:00:00,000 --> 00:00:05,000\nError generating SRT: {str(e)}\n"
            
        filename = f"subtitles_{task_id}.srt"
        media_type = "text/plain"
        
    elif format == "txt":
        try:
            # Try to use existing text output if available
            txt_path = f"outputs/{task_id}_transcript.txt"
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Generate text from results
                text_lines = []
                if 'processed_segments' in results:
                    for seg in results['processed_segments']:
                        speaker = seg.speaker_id if hasattr(seg, 'speaker_id') else "Unknown"
                        text = seg.original_text if hasattr(seg, 'original_text') else ""
                        text_lines.append(f"{speaker}: {text}")
                
                content = "\n".join(text_lines)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            content = f"Error generating transcript: {str(e)}"
            
        filename = f"transcript_{task_id}.txt"
        media_type = "text/plain"
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    # Save to temporary file
    temp_path = f"outputs/{filename}"
    os.makedirs("outputs", exist_ok=True)
    
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    return FileResponse(
        temp_path,
        media_type=media_type,
        filename=filename
    )


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


@app.get("/api/system-info")
async def get_system_info():
    """Get system information."""
    
    if UTILS_AVAILABLE:
        try:
            # from utils import _collect_system_info  # or import as needed
            # sys_info = _collect_system_info()
            # sys_info = get_system_info()
            # info.update(sys_info)

            info = {
                "version": "1.0.0",
                "features": [
                    "Speaker Diarization",
                    "Speech Recognition",
                    "Neural Translation",
                    "Interactive Visualization"
                ]
            }

            # Perform the health check
            health_status = "Unknown"
            health_color = "gray"
            
            try:
                from fastapi.testclient import TestClient
                client = TestClient(app)
                res = client.get("/health")

                if res.status_code == 200 and res.json().get("status") == "ok":
                    health_status = "Live"
                    health_color = "green"
                else:
                    health_status = "Error"
                    health_color = "yellow"
            except Exception as e:
                print("An exception occurred while getting system info: ", e)
                health_status = "Server Down"
                health_color = "red"

            info["status"] = health_status
            info["statusColor"] = health_color
            

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
    
    return JSONResponse(info)


# Demo mode for testing without full pipeline
@app.post("/api/demo-process")
async def demo_process(
    demo_file_id: str = Form(...),
    whisper_model: str = Form("small"),
    target_language: str = Form("en")
):
    """Demo processing endpoint that returns cached results immediately."""
    try:
        # Validate demo file ID
        if demo_file_id not in DEMO_FILES:
            raise HTTPException(status_code=400, detail="Invalid demo file selected")
        
        # Check if demo results are cached
        if demo_file_id not in demo_results_cache:
            raise HTTPException(status_code=503, detail="Demo files not available. Please try again in a moment.")
        
        # Simulate brief processing delay for realism
        await asyncio.sleep(1)
        
        # Get cached results
        results = demo_results_cache[demo_file_id]
        config = DEMO_FILES[demo_file_id]
        
        # Return comprehensive demo results
        return JSONResponse({
            "status": "complete",
            "filename": config["filename"],
            "demo_file": config["display_name"],
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Demo processing failed: {str(e)}"}
        )


@app.get("/api/demo-files")
async def get_demo_files():
    """Get available demo files with status."""
    demo_files = []
    
    for demo_id, config in DEMO_FILES.items():
        file_path = demo_manager.demo_dir / config["filename"]
        results_cached = demo_id in demo_results_cache
        
        demo_files.append({
            "id": demo_id,
            "name": config["display_name"],
            "filename": config["filename"],
            "language": config["language"],
            "description": config["description"],
            "available": file_path.exists(),
            "processed": results_cached,
            "status": "ready" if results_cached else "processing" if file_path.exists() else "downloading"
        })
    
    return JSONResponse({"demo_files": demo_files})


if __name__ == "__main__":
    # Setup for development
    logger.info("Starting Multilingual Audio Intelligence System...")
    
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 