#!/usr/bin/env python3
"""
Model Preloader for Multilingual Audio Intelligence System

This module handles downloading and initializing all AI models before the application starts.
It provides progress tracking, caching, and error handling for model loading.

Models loaded:
- pyannote.audio for speaker diarization
- faster-whisper for speech recognition
- mBART50 for neural machine translation
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

# Core imports
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
import psutil

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

class ModelPreloader:
    """Comprehensive model preloader with progress tracking and caching."""
    
    def __init__(self, cache_dir: str = "./model_cache", device: str = "auto"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.models = {}
        self.model_info = {}
        
        # Model configurations
        self.model_configs = {
            "speaker_diarization": {
                "name": "pyannote/speaker-diarization-3.1",
                "type": "pyannote",
                "description": "Speaker Diarization Pipeline",
                "size_mb": 32
            },
            "whisper_small": {
                "name": "small",
                "type": "whisper", 
                "description": "Whisper Speech Recognition (Small)",
                "size_mb": 484
            },
            "mbart_translation": {
                "name": "facebook/mbart-large-50-many-to-many-mmt",
                "type": "mbart",
                "description": "mBART Neural Machine Translation",
                "size_mb": 2440
            },
            "opus_mt_ja_en": {
                "name": "Helsinki-NLP/opus-mt-ja-en",
                "type": "opus_mt",
                "description": "Japanese to English Translation",
                "size_mb": 303
            },
            "opus_mt_es_en": {
                "name": "Helsinki-NLP/opus-mt-es-en", 
                "type": "opus_mt",
                "description": "Spanish to English Translation",
                "size_mb": 303
            },
            "opus_mt_fr_en": {
                "name": "Helsinki-NLP/opus-mt-fr-en",
                "type": "opus_mt", 
                "description": "French to English Translation",
                "size_mb": 303
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimal model loading."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "device": self.device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    def check_model_cache(self, model_key: str) -> bool:
        """Check if model is already cached and working."""
        cache_file = self.cache_dir / f"{model_key}_info.json"
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                cache_info = json.load(f)
                
            # Check if cache is recent (within 7 days)
            cache_time = datetime.fromisoformat(cache_info['timestamp'])
            days_old = (datetime.now() - cache_time).days
            
            if days_old > 7:
                logger.info(f"Cache for {model_key} is {days_old} days old, will refresh")
                return False
                
            return cache_info.get('status') == 'success'
        except Exception as e:
            logger.warning(f"Error reading cache for {model_key}: {e}")
            return False
    
    def save_model_cache(self, model_key: str, status: str, info: Dict[str, Any]):
        """Save model loading information to cache."""
        cache_file = self.cache_dir / f"{model_key}_info.json"
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "device": self.device,
            "info": info
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache for {model_key}: {e}")
    
    def load_pyannote_pipeline(self, task_id: str) -> Optional[Pipeline]:
        """Load pyannote speaker diarization pipeline."""
        try:
            console.print(f"[yellow]Loading pyannote.audio pipeline...[/yellow]")
            
            # Check for HuggingFace token
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                console.print("[red]Warning: HUGGINGFACE_TOKEN not found. Some models may not be accessible.[/red]")
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Test the pipeline
            console.print(f"[green]✓ pyannote.audio pipeline loaded successfully on {self.device}[/green]")
            
            return pipeline
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load pyannote.audio pipeline: {e}[/red]")
            logger.error(f"Pyannote loading failed: {e}")
            return None
    
    def load_whisper_model(self, task_id: str) -> Optional[WhisperModel]:
        """Load Whisper speech recognition model."""
        try:
            console.print(f"[yellow]Loading Whisper model (small)...[/yellow]")
            
            # Determine compute type based on device
            compute_type = "int8" if self.device == "cpu" else "float16"
            
            model = WhisperModel(
                "small",
                device=self.device,
                compute_type=compute_type,
                download_root=str(self.cache_dir / "whisper")
            )
            
            # Test the model with a dummy audio array
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            segments, info = model.transcribe(dummy_audio, language="en")
            list(segments)  # Force evaluation
            
            console.print(f"[green]✓ Whisper model loaded successfully on {self.device} with {compute_type}[/green]")
            
            return model
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load Whisper model: {e}[/red]")
            logger.error(f"Whisper loading failed: {e}")
            return None
    
    def load_mbart_model(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load mBART translation model."""
        try:
            console.print(f"[yellow]Loading mBART translation model...[/yellow]")
            
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            cache_path = self.cache_dir / "mbart"
            cache_path.mkdir(exist_ok=True)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_path)
            )
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(cache_path),
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Test the model
            test_input = tokenizer("Hello world", return_tensors="pt")
            if self.device != "cpu":
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                output = model.generate(**test_input, max_length=10)
            
            console.print(f"[green]✓ mBART model loaded successfully on {self.device}[/green]")
            
            return {
                "model": model,
                "tokenizer": tokenizer
            }
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load mBART model: {e}[/red]")
            logger.error(f"mBART loading failed: {e}")
            return None
    
    def load_opus_mt_model(self, task_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Load Opus-MT translation model."""
        try:
            console.print(f"[yellow]Loading Opus-MT model: {model_name}...[/yellow]")
            
            cache_path = self.cache_dir / "opus_mt" / model_name.replace("/", "--")
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_path)
            )
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=str(cache_path),
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Test the model
            test_input = tokenizer("Hello world", return_tensors="pt")
            if self.device != "cpu":
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                output = model.generate(**test_input, max_length=10)
            
            console.print(f"[green]✓ {model_name} loaded successfully on {self.device}[/green]")
            
            return {
                "model": model,
                "tokenizer": tokenizer
            }
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load {model_name}: {e}[/red]")
            logger.error(f"Opus-MT loading failed: {e}")
            return None
    
    def preload_all_models(self) -> Dict[str, Any]:
        """Preload all models with progress tracking."""
        
        # Display system information
        sys_info = self.get_system_info()
        
        info_panel = Panel.fit(
            f"""🖥️  System Information
            
• CPU Cores: {sys_info['cpu_count']}
• Total Memory: {sys_info['memory_gb']} GB
• Available Memory: {sys_info['available_memory_gb']} GB
• Device: {sys_info['device'].upper()}
• PyTorch: {sys_info['torch_version']}
• CUDA Available: {sys_info['cuda_available']}
{f"• GPU: {sys_info['gpu_name']}" if sys_info['gpu_name'] else ""}""",
            title="[bold blue]Audio Intelligence System[/bold blue]",
            border_style="blue"
        )
        console.print(info_panel)
        console.print()
        
        results = {
            "system_info": sys_info,
            "models": {},
            "total_time": 0,
            "success_count": 0,
            "total_count": len(self.model_configs)
        }
        
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Main progress bar
            main_task = progress.add_task("[cyan]Loading AI Models...", total=len(self.model_configs))
            
            # Load each model
            for model_key, config in self.model_configs.items():
                task_id = progress.add_task(f"[yellow]{config['description']}", total=100)
                
                # Check cache first
                if self.check_model_cache(model_key):
                    console.print(f"[green]✓ {config['description']} found in cache[/green]")
                    progress.update(task_id, completed=100)
                    progress.update(main_task, advance=1)
                    results["models"][model_key] = {"status": "cached", "time": 0}
                    results["success_count"] += 1
                    continue
                
                model_start_time = time.time()
                progress.update(task_id, completed=10)
                
                # Load model based on type
                if config["type"] == "pyannote":
                    model = self.load_pyannote_pipeline(task_id)
                elif config["type"] == "whisper":
                    model = self.load_whisper_model(task_id)
                elif config["type"] == "mbart":
                    model = self.load_mbart_model(task_id)
                elif config["type"] == "opus_mt":
                    model = self.load_opus_mt_model(task_id, config["name"])
                else:
                    model = None
                
                model_time = time.time() - model_start_time
                
                if model is not None:
                    self.models[model_key] = model
                    progress.update(task_id, completed=100)
                    results["models"][model_key] = {"status": "success", "time": model_time}
                    results["success_count"] += 1
                    
                    # Save to cache
                    self.save_model_cache(model_key, "success", {
                        "load_time": model_time,
                        "device": self.device,
                        "model_name": config["name"]
                    })
                else:
                    progress.update(task_id, completed=100)
                    results["models"][model_key] = {"status": "failed", "time": model_time}
                    
                    # Save failed status to cache
                    self.save_model_cache(model_key, "failed", {
                        "load_time": model_time,
                        "device": self.device,
                        "error": "Model loading failed"
                    })
                
                progress.update(main_task, advance=1)
        
        results["total_time"] = time.time() - start_time
        
        # Summary
        console.print()
        if results["success_count"] == results["total_count"]:
            status_text = "[bold green]✓ All models loaded successfully![/bold green]"
            status_color = "green"
        elif results["success_count"] > 0:
            status_text = f"[bold yellow]⚠ {results['success_count']}/{results['total_count']} models loaded[/bold yellow]"
            status_color = "yellow"
        else:
            status_text = "[bold red]✗ No models loaded successfully[/bold red]"
            status_color = "red"
        
        summary_panel = Panel.fit(
            f"""{status_text}

• Loading Time: {results['total_time']:.1f} seconds
• Device: {self.device.upper()}
• Memory Usage: {psutil.virtual_memory().percent:.1f}%
• Models Ready: {results['success_count']}/{results['total_count']}""",
            title="[bold]Model Loading Summary[/bold]",
            border_style=status_color
        )
        console.print(summary_panel)
        
        return results
    
    def get_models(self) -> Dict[str, Any]:
        """Get loaded models."""
        return self.models
    
    def cleanup(self):
        """Cleanup resources."""
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main function to run model preloading."""
    console.print(Panel.fit(
        "[bold blue]🎵 Multilingual Audio Intelligence System[/bold blue]\n[yellow]Model Preloader[/yellow]",
        border_style="blue"
    ))
    console.print()
    
    # Initialize preloader
    preloader = ModelPreloader()
    
    # Load all models
    try:
        results = preloader.preload_all_models()
        
        if results["success_count"] > 0:
            console.print("\n[bold green]✓ Model preloading completed![/bold green]")
            console.print(f"[dim]Models cached in: {preloader.cache_dir}[/dim]")
            return True
        else:
            console.print("\n[bold red]✗ Model preloading failed![/bold red]")
            return False
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Model preloading interrupted by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✗ Model preloading failed: {e}[/bold red]")
        logger.error(f"Preloading failed: {e}")
        return False
    finally:
        preloader.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 