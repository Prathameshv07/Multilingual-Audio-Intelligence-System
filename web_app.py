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
import secrets
from collections import defaultdict

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

# Load environment variables with error handling
try:
    load_dotenv()
except Exception as e:
    logging.warning(f"Could not load .env file: {e}")

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
    from src.main import AudioIntelligencePipeline
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

# ENHANCED Demo file configuration with NEW Indian Language Support
DEMO_FILES = {
    "yuri_kizaki": {
        "name": "Yuri Kizaki",
        "filename": "Yuri_Kizaki.mp3",
        "display_name": "🇯🇵 Japanese Business Communication",
        "language": "ja",
        "description": "Professional audio message about website communication and business enhancement",
        "url": "https://www.mitsue.co.jp/service/audio_and_video/audio_production/media/narrators_sample/yuri_kizaki/03.mp3",
        "expected_text": "音声メッセージが既存のウェブサイトを超えたコミュニケーションを実現。目で見るだけだったウェブサイトに音声情報をインクルードすることで、情報に新しい価値を与え、他者との差別化に効果を発揮します。",
        "expected_translation": "Audio messages enable communication beyond existing websites. By incorporating audio information into visually-driven websites, you can add new value to the information and effectively differentiate your website from others.",
        "category": "business",
        "difficulty": "intermediate",
        "duration": "00:00:32"
    },
    "film_podcast": {
        "name": "Film Podcast",
        "filename": "Film_Podcast.mp3", 
        "display_name": "🇫🇷 French Cinema Discussion",
        "language": "fr",
        "description": "In-depth French podcast discussing recent movies including Social Network and Paranormal Activity",
        "url": "https://www.lightbulblanguages.co.uk/resources/audio/film-podcast.mp3",
        "expected_text": "Le film intitulé The Social Network traite de la création du site Facebook par Mark Zuckerberg et des problèmes judiciaires que cela a comporté pour le créateur de ce site.",
        "expected_translation": "The film The Social Network deals with the creation of Facebook by Mark Zuckerberg and the legal problems this caused for the creator of this site.",
        "category": "entertainment",
        "difficulty": "advanced",
        "duration": "00:03:50"
    },
    "tamil_interview": {
        "name": "Tamil Wikipedia Interview",
        "filename": "Tamil_Wikipedia_Interview.ogg",
        "display_name": "🇮🇳 Tamil Wikipedia Interview",
        "language": "ta",
        "description": "NEW: Tamil language interview about Wikipedia and collaborative knowledge sharing in South India",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/54/Parvathisri-Wikipedia-Interview-Vanavil-fm.ogg",
        "expected_text": "விக்கிபீடியா என்பது ஒரு கூட்டு முயற்சியாகும். இது தமிழ் மொழியில் அறிவைப் பகிர்ந்து கொள்வதற்கான ஒரு சிறந்த தளமாகும்.",
        "expected_translation": "Wikipedia is a collaborative effort. It is an excellent platform for sharing knowledge in the Tamil language.",
        "category": "education",
        "difficulty": "advanced",
        "duration": "00:36:17",
        "featured": True,
        "new": True,
        "indian_language": True
    },
    "car_trouble": {
        "name": "Car Trouble",
        "filename": "Car_Trouble.mp3",
        "display_name": "🇮🇳 Hindi Daily Conversation",
        "language": "hi", 
        "description": "NEW: Real-world Hindi conversation about car problems and waiting for a mechanic",
        "url": "https://www.tuttlepublishing.com/content/docs/9780804844383/06-18%20Part2%20Car%20Trouble.mp3",
        "expected_text": "गाड़ी खराब हो गई है। मैकेनिक का इंतज़ार कर रहे हैं। कुछ समय लगेगा।",
        "expected_translation": "The car has broken down. We are waiting for the mechanic. It will take some time.",
        "category": "daily_life",
        "difficulty": "beginner", 
        "duration": "00:00:45",
        "featured": True,
        "new": True,
        "indian_language": True
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

# Session management
user_sessions = defaultdict(dict)
session_files = defaultdict(list)

def transform_to_old_format(results):
    """Transform new JSON format to old format expected by frontend."""
    try:
        # If it's already in old format, return as-is
        if 'segments' in results and 'summary' in results:
            return results
        
        # Transform new format to old format
        segments = []
        summary = {}
        
        # Try to extract segments from different possible locations
        if 'outputs' in results and 'json' in results['outputs']:
            # Parse the JSON string in outputs.json
            try:
                parsed_outputs = json.loads(results['outputs']['json'])
                if 'segments' in parsed_outputs:
                    segments = parsed_outputs['segments']
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback: try direct segments
        if not segments and 'segments' in results:
            segments = results['segments']
        
        # Build summary from processing_stats
        if 'processing_stats' in results:
            stats = results['processing_stats']
            summary = {
                'total_duration': results.get('audio_metadata', {}).get('duration_seconds', 0),
                'num_speakers': stats.get('num_speakers', 1),
                'num_segments': stats.get('num_segments', len(segments)),
                'languages': stats.get('languages_detected', ['unknown']),
                'processing_time': stats.get('total_time', 0)
            }
        else:
            # Fallback summary
            summary = {
                'total_duration': 0,
                'num_speakers': 1,
                'num_segments': len(segments),
                'languages': ['unknown'],
                'processing_time': 0
            }
        
        # Ensure segments have the correct format
        formatted_segments = []
        for seg in segments:
            if isinstance(seg, dict):
                formatted_seg = {
                    'speaker': seg.get('speaker_id', seg.get('speaker', 'SPEAKER_00')),
                    'start_time': seg.get('start_time', 0),
                    'end_time': seg.get('end_time', 0),
                    'text': seg.get('original_text', seg.get('text', '')),
                    'translated_text': seg.get('translated_text', ''),
                    'language': seg.get('original_language', seg.get('language', 'unknown'))
                }
                formatted_segments.append(formatted_seg)
        
        result = {
            'segments': formatted_segments,
            'summary': summary
        }
        
        logger.info(f"✅ Transformed results: {len(formatted_segments)} segments, summary keys: {list(summary.keys())}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error transforming results to old format: {e}")
        # Return minimal fallback structure
        return {
            'segments': [],
            'summary': {
                'total_duration': 0,
                'num_speakers': 0,
                'num_segments': 0,
                'languages': [],
                'processing_time': 0
            }
        }

class SessionManager:
    """Manages user sessions and cleanup."""
    
    def __init__(self):
        self.sessions = user_sessions
        self.session_files = session_files
        self.cleanup_interval = 3600  # 1 hour
        
    def generate_session_id(self, request: Request) -> str:
        """Generate a unique session ID based on user fingerprint."""
        # Create a stable fingerprint from IP and user agent (no randomness for consistency)
        fingerprint_data = [
            request.client.host if request.client else "unknown",
            request.headers.get("user-agent", "")[:100],  # Truncate for consistency
            request.headers.get("accept-language", "")[:50],  # Truncate for consistency
        ]
        
        # Create hash (no randomness so same user gets same session)
        fingerprint = "|".join(fingerprint_data)
        session_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        
        # Initialize session if new
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": time.time(),
                "last_activity": time.time(),
                "ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "")[:100]  # Truncate for storage
            }
            logger.info(f"🔑 New session created: {session_id}")
        else:
            # Update last activity
            self.sessions[session_id]["last_activity"] = time.time()
            
        return session_id
    
    def add_file_to_session(self, session_id: str, file_path: str):
        """Associate a file with a user session."""
        self.session_files[session_id].append({
            "file_path": file_path,
            "created_at": time.time()
        })
        logger.info(f"📁 Added file to session {session_id}: {file_path}")
    
    def cleanup_session(self, session_id: str):
        """Clean up all files associated with a session."""
        if session_id not in self.session_files:
            return
            
        files_cleaned = 0
        for file_info in self.session_files[session_id]:
            file_path = Path(file_info["file_path"])
            try:
                if file_path.exists():
                    file_path.unlink()
                    files_cleaned += 1
                    logger.info(f"🗑️ Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
        
        # Clean up session data
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_files:
            del self.session_files[session_id]
            
        logger.info(f"✅ Session cleanup completed for {session_id}: {files_cleaned} files removed")
        return files_cleaned
    
    def cleanup_expired_sessions(self):
        """Clean up sessions that haven't been active for a while."""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in list(self.sessions.items()):
            if current_time - session_data["last_activity"] > self.cleanup_interval:
                expired_sessions.append(session_id)
        
        total_cleaned = 0
        for session_id in expired_sessions:
            files_cleaned = self.cleanup_session(session_id)
            total_cleaned += files_cleaned
            
        if expired_sessions:
            logger.info(f"🕒 Expired session cleanup: {len(expired_sessions)} sessions, {total_cleaned} files")
        
        return len(expired_sessions), total_cleaned

# Initialize session manager
session_manager = SessionManager()

class DemoManager:
    """Manages demo files and preprocessing."""
    
    def __init__(self):
        self.demo_dir = Path("demo_audio")
        self.demo_dir.mkdir(exist_ok=True)
        self.results_dir = Path("demo_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def ensure_demo_files(self):
        """Ensure demo files are available and processed."""
        logger.info("🔄 Checking demo files...")
        
        for demo_id, config in DEMO_FILES.items():
            logger.info(f"📁 Checking demo file: {config['filename']}")
            file_path = self.demo_dir / config["filename"]
            results_path = self.results_dir / f"{demo_id}_results.json"
            
            # Check if file exists, download if not
            if not file_path.exists():
                if config["url"] == "local":
                    logger.warning(f"❌ Local demo file not found: {config['filename']}")
                    logger.info(f"   Expected location: {file_path}")
                    continue
                else:
                    logger.info(f"⬇️ Downloading demo file: {config['filename']}")
                    try:
                        await self.download_demo_file(config["url"], file_path)
                        logger.info(f"✅ Downloaded: {config['filename']}")
                    except Exception as e:
                        logger.error(f"❌ Failed to download {config['filename']}: {e}")
                        continue
            else:
                logger.info(f"✅ Demo file exists: {config['filename']}")
            
            # Check if results exist, process if not
            if not results_path.exists():
                logger.info(f"🔄 Processing demo file: {config['filename']} (first time)")
                try:
                    await self.process_demo_file(demo_id, file_path, results_path)
                    logger.info(f"✅ Demo processing completed: {config['filename']}")
                except Exception as e:
                    logger.error(f"❌ Failed to process {config['filename']}: {e}")
                    continue
            else:
                logger.info(f"📋 Using cached results: {demo_id}")
            
            # Load results into cache
            try:
                if results_path.exists() and results_path.stat().st_size > 0:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        demo_results_cache[demo_id] = json.load(f)
                    logger.info(f"✅ Loaded cached results for {demo_id}")
                else:
                    logger.warning(f"⚠️ Results file empty or missing for {demo_id}")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in {demo_id} results: {e}")
                # Delete corrupted file and reprocess
                if results_path.exists():
                    results_path.unlink()
                    logger.info(f"🗑️ Deleted corrupted results for {demo_id}, will reprocess on next startup")
            except Exception as e:
                logger.error(f"❌ Failed to load cached results for {demo_id}: {e}")
        
        logger.info(f"✅ Demo files check completed. Available: {len(demo_results_cache)}")
    
    async def download_demo_file(self, url: str, file_path: Path):
        """Download demo file from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded demo file: {file_path.name}")
    
    async def process_demo_file(self, demo_id: str, file_path: Path, results_path: Path):
        """Process a demo file and cache results."""
        logger.info(f"🎵 Starting demo processing: {file_path.name}")
        
        try:
            # Use the global pipeline instance
            global pipeline
            if pipeline is None:
                from src.main import AudioIntelligencePipeline
                pipeline = AudioIntelligencePipeline(
                    whisper_model_size="small",
                    target_language="en",
                    device="cpu"
                )
            
            # Process the audio file
            results = pipeline.process_audio(
                audio_file=file_path,
                output_dir=Path("outputs")
            )
            
            # Save results to cache file
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Store in memory cache
            demo_results_cache[demo_id] = results
            
            logger.info(f"✅ Demo processing completed and cached: {file_path.name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Demo processing failed for {file_path.name}: {e}")
            raise
    
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




        
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse("index.html", {"request": request})
            
        
@app.post("/api/upload")
async def upload_audio(
    request: Request,
            file: UploadFile = File(...),
    whisper_model: str = Form("small"),
    target_language: str = Form("en"),
    hf_token: Optional[str] = Form(None)
        ):
            """Upload and process audio file."""
            try:
                # Generate session ID for this user
                session_id = session_manager.generate_session_id(request)
                logger.info(f"🔑 Processing upload for session: {session_id}")
                
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
                
                # Save uploaded file with session ID
                file_path = f"uploads/{session_id}_{int(time.time())}_{file.filename}"
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Track file in session
                session_manager.add_file_to_session(session_id, file_path)
                
                # Generate task ID with session
                task_id = f"task_{session_id}_{int(time.time())}"
        
                # Start background processing
                asyncio.create_task(
                audio_processor.process_audio_file(
                    file_path, whisper_model, target_language, hf_token, task_id
                ))
                            
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
        logger.info(f"📊 Found results for task {task_id}: {type(results)}")
        logger.info(f"📊 Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")

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
            languages = set(
                seg["language"] for seg in formatted_results["segments"] if seg["language"] != 'unknown'
            )
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

        logger.info(f"📤 Returning formatted results for task {task_id}: {len(formatted_results.get('segments', []))} segments")
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


# async def get_results(task_id: str):
#     """Get processing results."""
#     if task_id not in processing_status:
#         raise HTTPException(status_code=404, detail="Task not found")
    
#     status = processing_status[task_id]
#     if status.get("status") != "complete":
#         raise HTTPException(status_code=202, detail="Processing not complete")
    
#     # Return actual processed results
#     if task_id in processing_results:
#         results = processing_results[task_id]
        
#         # Convert to the expected format for frontend
#         formatted_results = {
#             "segments": [],
#             "summary": {
#                 "total_duration": 0,
#                 "num_speakers": 0,
#                 "num_segments": 0,
#                 "languages": [],
#                 "processing_time": 0
#             }
#         }
        
#         try:
#             # Extract segments information
#             if 'processed_segments' in results:
#                 for seg in results['processed_segments']:
#                     formatted_results["segments"].append({
#                         "speaker": seg.speaker_id if hasattr(seg, 'speaker_id') else "Unknown Speaker",
#                         "start_time": seg.start_time if hasattr(seg, 'start_time') else 0,
#                         "end_time": seg.end_time if hasattr(seg, 'end_time') else 0,
#                         "text": seg.original_text if hasattr(seg, 'original_text') else "",
#                         "translated_text": seg.translated_text if hasattr(seg, 'translated_text') else "",
#                         "language": seg.original_language if hasattr(seg, 'original_language') else "unknown",
#                     })
            
#             # Extract summary information
#             if 'audio_metadata' in results:
#                 metadata = results['audio_metadata']
#                 formatted_results["summary"]["total_duration"] = metadata.get('duration_seconds', 0)
            
#             if 'processing_stats' in results:
#                 stats = results['processing_stats']
#                 formatted_results["summary"]["processing_time"] = stats.get('total_time', 0)
            
#             # Calculate derived statistics
#             formatted_results["summary"]["num_segments"] = len(formatted_results["segments"])
#             speakers = set(seg["speaker"] for seg in formatted_results["segments"])
#             formatted_results["summary"]["num_speakers"] = len(speakers)
#             languages = set(seg["language"] for seg in formatted_results["segments"] if seg["language"] != 'unknown')
#             formatted_results["summary"]["languages"] = list(languages) if languages else ["unknown"]
                
#         except Exception as e:
#             logger.error(f"Error formatting results: {e}")
#             # Fallback to basic structure
#             formatted_results = {
#                 "segments": [
#                     {
#                         "speaker": "Speaker 1",
#                         "start_time": 0.0,
#                         "end_time": 5.0,
#                         "text": f"Processed audio from file. Full results processing encountered an error: {str(e)}",
#                         "language": "en",
#                     }
#                 ],
#                 "summary": {
#                     "total_duration": 5.0,
#                     "num_speakers": 1,
#                     "num_segments": 1,
#                     "languages": ["en"],
#                     "processing_time": 2.0
#                 }
#             }
        
#         return JSONResponse({
#             "task_id": task_id,
#             "status": "complete",
#             "results": formatted_results
#         })
#     else:
#         # Fallback if results not found
#         return JSONResponse({
#             "task_id": task_id,
#             "status": "complete",
#             "results": {
#                 "segments": [
#                     {
#                         "speaker": "System",
#                         "start_time": 0.0,
#                         "end_time": 1.0,
#                         "text": "Audio processing completed but results are not available for display.",
#                         "language": "en",
#                     }
#                 ],
#                 "summary": {
#                     "total_duration": 1.0,
#                     "num_speakers": 1,
#                     "num_segments": 1,
#                     "languages": ["en"],
#                     "processing_time": 0.1
#                 }
#             }
#         })


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
    
    # Initialize default info
    info = {
        "version": "1.0.0",
        "features": [
            "Speaker Diarization",
            "Speech Recognition", 
            "Neural Translation",
            "Interactive Visualization"
        ],
        "status": "Live",
        "statusColor": "green"
    }
    
    if UTILS_AVAILABLE:
        try:
            # Enhanced system info collection when utils are available

            # Simple health check without httpx dependency issues
            health_status = "Live"
            health_color = "green"
            
            # Add system information
            import psutil
            import platform
            
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                info.update({
                    "system": {
                        "platform": platform.system(),
                        "python_version": platform.python_version(),
                        "cpu_usage": f"{cpu_percent}%",
                        "memory_usage": f"{memory.percent}%",
                        "disk_usage": f"{disk.percent}%"
                    }
                })
            except ImportError:
                # If psutil is not available, just show basic info
                info.update({
                    "system": {
                        "platform": platform.system(),
                        "python_version": platform.python_version()
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
            
            info["status"] = health_status
            info["statusColor"] = health_color
            

        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
    
    return JSONResponse(info)


# Note: Old demo-process endpoint removed in favor of process-demo/{demo_id}


@app.get("/api/demo-files")
async def get_demo_files():
    """Get available demo files with status."""
    try:
        demo_files = []
        
        logger.info(f"📋 Building demo files list from {len(DEMO_FILES)} configurations")
        
        for demo_id, config in DEMO_FILES.items():
            file_path = demo_manager.demo_dir / config["filename"]
            results_cached = demo_id in demo_results_cache
            
            demo_file_info = {
                "id": demo_id,
                "name": config.get("name", config.get("display_name", demo_id)),
                "filename": config["filename"],
                "language": config["language"],
                "description": config["description"],
                "category": config.get("category", "general"),
                "difficulty": config.get("difficulty", "intermediate"),
                "duration": config.get("duration", "unknown"),
                "featured": config.get("featured", False),
                "new": config.get("new", False),
                "indian_language": config.get("indian_language", False),
                "available": file_path.exists(),
                "processed": results_cached,
                "status": "ready" if results_cached else "processing" if file_path.exists() else "downloading"
            }
            
            demo_files.append(demo_file_info)
            logger.info(f"📁 Added demo file: {demo_id} -> {demo_file_info['name']}")
        
        logger.info(f"✅ Returning {len(demo_files)} demo files to frontend")
        return JSONResponse(demo_files)
        
    except Exception as e:
        logger.error(f"❌ Error building demo files list: {e}")
        return JSONResponse({"demo_files": [], "error": str(e)})


@app.get("/demo_audio/{filename}")
async def get_demo_audio(filename: str):
    """Serve demo audio files."""
    try:
        # Security: prevent path traversal
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Check if file exists in demo_audio directory
        audio_path = Path("demo_audio") / filename
        if not audio_path.exists():
            # Try with common extensions
            for ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                audio_path_with_ext = Path("demo_audio") / f"{filename}{ext}"
                if audio_path_with_ext.exists():
                    audio_path = audio_path_with_ext
                    break
            else:
                raise HTTPException(status_code=404, detail="Demo audio file not found")
        
        # Determine content type
        content_type = "audio/mpeg"  # default
        if audio_path.suffix.lower() == '.ogg':
            content_type = "audio/ogg"
        elif audio_path.suffix.lower() == '.wav':
            content_type = "audio/wav"
        elif audio_path.suffix.lower() == '.m4a':
            content_type = "audio/mp4"
        
        logger.info(f"📻 Serving demo audio: {audio_path}")
        return FileResponse(
            path=str(audio_path),
            media_type=content_type,
            filename=audio_path.name
        )
        
    except Exception as e:
        logger.error(f"Error serving demo audio {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve demo audio")


@app.post("/api/process-demo/{demo_id}")
async def process_demo_by_id(demo_id: str):
    """Process demo file by ID and return cached results."""
    try:
        logger.info(f"🎯 Processing demo file: {demo_id}")
        
        # Check if demo file exists
        if demo_id not in DEMO_FILES:
            raise HTTPException(status_code=404, detail=f"Demo file '{demo_id}' not found")
        
        # Check if results are cached
        results_path = Path("demo_results") / f"{demo_id}_results.json"
        
        if results_path.exists():
            logger.info(f"📁 Loading cached results for {demo_id}")
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Transform new format to old format if needed
                transformed_results = transform_to_old_format(results)
                
                return JSONResponse({
                    "status": "complete",
                    "results": transformed_results
                })
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse cached results for {demo_id}: {e}")
                # Fall through to reprocess
        
        # If not cached, process the demo file
        logger.info(f"⚡ Processing demo file {demo_id} on-demand")
        file_path = demo_manager.demo_dir / DEMO_FILES[demo_id]["filename"]
        
        if not file_path.exists():
            # Try to download the file first
            try:
                config = DEMO_FILES[demo_id]
                await demo_manager.download_demo_file(config["url"], file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to download demo file: {str(e)}")
        
        # Process the file
        results = await demo_manager.process_demo_file(demo_id, file_path, results_path)
        
        # Transform new format to old format
        transformed_results = transform_to_old_format(results)
        
        return JSONResponse({
            "status": "complete", 
            "results": transformed_results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing demo {demo_id}: {e}")
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)


@app.post("/api/cleanup")
async def cleanup_session(request: Request):
    """Clean up user session files."""
    try:
        session_id = session_manager.generate_session_id(request)
        files_cleaned = session_manager.cleanup_session(session_id)
        
        return JSONResponse({
            "status": "success",
            "message": f"Cleaned up {files_cleaned} files for session {session_id}",
            "files_cleaned": files_cleaned
        })
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Cleanup failed: {str(e)}"}
        )


@app.post("/api/cleanup-expired")
async def cleanup_expired():
    """Clean up expired sessions (admin endpoint)."""
    try:
        sessions_cleaned, files_cleaned = session_manager.cleanup_expired_sessions()
        
        return JSONResponse({
            "status": "success",
            "message": f"Cleaned up {sessions_cleaned} expired sessions",
            "sessions_cleaned": sessions_cleaned,
            "files_cleaned": files_cleaned
        })
        
    except Exception as e:
        logger.error(f"❌ Expired cleanup error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Expired cleanup failed: {str(e)}"}
        )


@app.get("/api/session-info")
async def get_session_info(request: Request):
    """Get current session information."""
    try:
        session_id = session_manager.generate_session_id(request)
        session_data = session_manager.sessions.get(session_id, {})
        files_count = len(session_manager.session_files.get(session_id, []))
        
        return JSONResponse({
            "session_id": session_id,
            "created_at": session_data.get("created_at"),
            "last_activity": session_data.get("last_activity"),
            "files_count": files_count,
            "status": "active"
        })
        
    except Exception as e:
        logger.error(f"❌ Session info error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Session info failed: {str(e)}"}
        )


async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 Starting Multilingual Audio Intelligence System...")
    try:
        system_info = get_system_info()
        logger.info(f"📊 System Info: {system_info}")
    except Exception as e:
        logger.warning(f"⚠️ Could not get system info: {e}")
        logger.info("📊 System Info: [System info unavailable]")
    
    # Initialize demo manager
    global demo_manager
    demo_manager = DemoManager()
    await demo_manager.ensure_demo_files()
    
    # Clean up any expired sessions on startup
    sessions_cleaned, files_cleaned = session_manager.cleanup_expired_sessions()
    if sessions_cleaned > 0:
        logger.info(f"🧹 Startup cleanup: {sessions_cleaned} expired sessions, {files_cleaned} files")
    
    logger.info("✅ Startup completed successfully!")

async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🛑 Shutting down Multilingual Audio Intelligence System...")
    
    # Clean up all active sessions on shutdown
    total_sessions = len(session_manager.sessions)
    total_files = 0
    for session_id in list(session_manager.sessions.keys()):
        files_cleaned = session_manager.cleanup_session(session_id)
        total_files += files_cleaned
    
    if total_sessions > 0:
        logger.info(f"🧹 Shutdown cleanup: {total_sessions} sessions, {total_files} files")

# Register startup and shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Enhanced logging for requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} → {response.status_code} ({process_time:.2f}s)")
    
    return response

if __name__ == "__main__":
    # Start server
        uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    ) 