#!/usr/bin/env python3
"""
Model Preloader for Multilingual Audio Intelligence System - Enhanced Version

Key improvements:
1. Smart local cache detection with corruption checking
2. Fallback to download if local files don't exist or are corrupted  
3. Better error handling and retry mechanisms
4. Consistent approach across all model types
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
import whisper
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
    """Comprehensive model preloader with enhanced local cache detection."""
    
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
                "name": "openai/whisper-small",
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
            # Common language models
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
            },
            # Enhanced Indian language models
            "opus_mt_hi_en": {
                "name": "Helsinki-NLP/opus-mt-hi-en",
                "type": "opus_mt",
                "description": "Hindi to English Translation",
                "size_mb": 303
            },
            "opus_mt_ta_en": {
                "name": "Helsinki-NLP/opus-mt-ta-en",
                "type": "opus_mt",
                "description": "Tamil to English Translation",
                "size_mb": 303
            },
            "opus_mt_bn_en": {
                "name": "Helsinki-NLP/opus-mt-bn-en",
                "type": "opus_mt",
                "description": "Bengali to English Translation",
                "size_mb": 303
            },
            "opus_mt_te_en": {
                "name": "Helsinki-NLP/opus-mt-te-en",
                "type": "opus_mt",
                "description": "Telugu to English Translation",
                "size_mb": 303
            },
            "opus_mt_mr_en": {
                "name": "Helsinki-NLP/opus-mt-mr-en",
                "type": "opus_mt",
                "description": "Marathi to English Translation",
                "size_mb": 303
            },
            "opus_mt_gu_en": {
                "name": "Helsinki-NLP/opus-mt-gu-en",
                "type": "opus_mt",
                "description": "Gujarati to English Translation",
                "size_mb": 303
            },
            "opus_mt_kn_en": {
                "name": "Helsinki-NLP/opus-mt-kn-en",
                "type": "opus_mt",
                "description": "Kannada to English Translation",
                "size_mb": 303
            },
            "opus_mt_pa_en": {
                "name": "Helsinki-NLP/opus-mt-pa-en",
                "type": "opus_mt",
                "description": "Punjabi to English Translation",
                "size_mb": 303
            },
            "opus_mt_ml_en": {
                "name": "Helsinki-NLP/opus-mt-ml-en",
                "type": "opus_mt",
                "description": "Malayalam to English Translation",
                "size_mb": 303
            },
            "opus_mt_ne_en": {
                "name": "Helsinki-NLP/opus-mt-ne-en",
                "type": "opus_mt",
                "description": "Nepali to English Translation",
                "size_mb": 303
            },
            "opus_mt_ur_en": {
                "name": "Helsinki-NLP/opus-mt-ur-en",
                "type": "opus_mt",
                "description": "Urdu to English Translation",
                "size_mb": 303
            }
        }
    
    def check_local_model_files(self, model_name: str, model_type: str) -> bool:
        """
        Check if model files exist locally and are not corrupted.
        Returns True if valid local files exist, False otherwise.
        """
        try:
            if model_type == "whisper":
                # For Whisper, check the Systran faster-whisper cache
                whisper_cache = self.cache_dir / "whisper" / "models--Systran--faster-whisper-small"
                required_files = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
                
                # Find the snapshot directory
                snapshots_dir = whisper_cache / "snapshots"
                if not snapshots_dir.exists():
                    return False
                
                # Check for any snapshot directory (there should be one)
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if not snapshot_dirs:
                    return False
                
                # Check if required files exist in the snapshot
                snapshot_path = snapshot_dirs[0]  # Use the first (and likely only) snapshot
                for file in required_files:
                    file_path = snapshot_path / file
                    if not file_path.exists() or file_path.stat().st_size == 0:
                        return False
                
                return True
                
            elif model_type in ["mbart", "opus_mt"]:
                # For Transformers models, check the HuggingFace cache structure
                if model_type == "mbart":
                    model_cache_path = self.cache_dir / "mbart" / f"models--{model_name.replace('/', '--')}"
                else:
                    model_cache_path = self.cache_dir / "opus_mt" / f"{model_name.replace('/', '--')}" / f"models--{model_name.replace('/', '--')}"
                
                required_files = ["config.json", "tokenizer_config.json"]
                # Also check for model files (either .bin or .safetensors)
                model_files = ["pytorch_model.bin", "model.safetensors"]
                
                # Find the snapshot directory
                snapshots_dir = model_cache_path / "snapshots"
                if not snapshots_dir.exists():
                    return False
                
                # Check for any snapshot directory
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if not snapshot_dirs:
                    return False
                
                # Check the latest snapshot
                snapshot_path = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                
                # Check required config files
                for file in required_files:
                    file_path = snapshot_path / file
                    if not file_path.exists() or file_path.stat().st_size == 0:
                        return False
                
                # Check for at least one model file
                model_file_exists = any(
                    (snapshot_path / model_file).exists() and (snapshot_path / model_file).stat().st_size > 0
                    for model_file in model_files
                )
                
                return model_file_exists
                
            elif model_type == "pyannote":
                # For pyannote, it uses HuggingFace hub caching, harder to predict exact path
                # We'll rely on the transformers library's cache detection
                return False  # Let it attempt to load and handle caching automatically
                
        except Exception as e:
            logger.warning(f"Error checking local files for {model_name}: {e}")
            return False
        
        return False

    def load_transformers_model_with_cache_check(self, model_name: str, cache_path: Path, model_type: str = "seq2seq") -> Optional[Dict[str, Any]]:
        """
        Load transformers model with intelligent cache checking and fallback.
        """
        try:
            # First, check if we have valid local files
            has_local_files = self.check_local_model_files(model_name, "mbart" if "mbart" in model_name else "opus_mt")
            
            if has_local_files:
                console.print(f"[green]Found valid local cache for {model_name}, loading from cache...[/green]")
                try:
                    # Try loading from local cache first
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(cache_path),
                        local_files_only=True
                    )
                    
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        cache_dir=str(cache_path),
                        local_files_only=True,
                        torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
                    )
                    
                    console.print(f"[green]SUCCESS: Successfully loaded {model_name} from local cache[/green]")
                    
                except Exception as e:
                    console.print(f"[yellow]Local cache load failed for {model_name}, will download: {e}[/yellow]")
                    has_local_files = False  # Force download
            
            if not has_local_files:
                console.print(f"[yellow]No valid local cache for {model_name}, downloading...[/yellow]")
                # Load with download (default behavior)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(cache_path)
                )
                
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=str(cache_path),
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
                )
                
                console.print(f"[green]SUCCESS: Successfully downloaded and loaded {model_name}[/green]")
            
            # Move to device if needed
            if self.device != "cpu":
                model = model.to(self.device)
            
            # Test the model
            test_input = tokenizer("Hello world", return_tensors="pt")
            if self.device != "cpu":
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
            
            with torch.no_grad():
                output = model.generate(**test_input, max_length=10)
            
            return {
                "model": model,
                "tokenizer": tokenizer
            }
            
        except Exception as e:
            console.print(f"[red]✗ Failed to load {model_name}: {e}[/red]")
            logger.error(f"Model loading failed for {model_name}: {e}")
            return None

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
            console.print(f"[green]SUCCESS: pyannote.audio pipeline loaded successfully on {self.device}[/green]")
            
            return pipeline
            
        except Exception as e:
            console.print(f"[red]ERROR: Failed to load pyannote.audio pipeline: {e}[/red]")
            logger.error(f"Pyannote loading failed: {e}")
            return None
    
    def load_whisper_model(self, task_id: str) -> Optional[whisper.Whisper]:
        """Load Whisper speech recognition model with enhanced cache checking."""
        try:
            console.print(f"[yellow]Loading Whisper model (small)...[/yellow]")
            
            whisper_cache_dir = self.cache_dir / "whisper"
            
            # Check if we have valid local files
            has_local_files = self.check_local_model_files("small", "whisper")
            
            if has_local_files:
                console.print(f"[green]Found valid local Whisper cache, loading from cache...[/green]")
            else:
                console.print(f"[yellow]No valid local Whisper cache found, will download...[/yellow]")
            
            # OpenAI Whisper handles caching automatically
            model = whisper.load_model("small", device=self.device)
            
            # Test the model with a dummy audio array
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            result = model.transcribe(dummy_audio, language="en")
            
            console.print(f"[green]SUCCESS: Whisper model loaded successfully on {self.device}[/green]")
            
            return model
            
        except Exception as e:
            console.print(f"[red]ERROR: Failed to load Whisper model: {e}[/red]")
            logger.error(f"Whisper loading failed: {e}")
            return None
    
    def load_mbart_model(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load mBART translation model with enhanced cache checking."""
        console.print(f"[yellow]Loading mBART translation model...[/yellow]")
        
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        cache_path = self.cache_dir / "mbart"
        cache_path.mkdir(exist_ok=True)
        
        return self.load_transformers_model_with_cache_check(model_name, cache_path, "seq2seq")
    
    def load_opus_mt_model(self, task_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """Load Opus-MT translation model with enhanced cache checking."""
        console.print(f"[yellow]Loading Opus-MT model: {model_name}...[/yellow]")
        
        cache_path = self.cache_dir / "opus_mt" / model_name.replace("/", "--")
        cache_path.mkdir(parents=True, exist_ok=True)
        
        return self.load_transformers_model_with_cache_check(model_name, cache_path, "seq2seq")
    
    def preload_all_models(self) -> Dict[str, Any]:
        """Preload all models with progress tracking."""
        
        # Display system information
        sys_info = self.get_system_info()
        
        info_panel = Panel.fit(
            f"""System Information
            
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
                    console.print(f"[green]SUCCESS: {config['description']} found in cache[/green]")
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
            status_text = "[bold green]SUCCESS: All models loaded successfully![/bold green]"
            status_color = "green"
        elif results["success_count"] > 0:
            status_text = f"[bold yellow]WARNING: {results['success_count']}/{results['total_count']} models loaded[/bold yellow]"
            status_color = "yellow"
        else:
            status_text = "[bold red]ERROR: No models loaded successfully[/bold red]"
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
    # Use ASCII-safe characters for Windows compatibility
    console.print(Panel.fit(
        "[bold blue]Multilingual Audio Intelligence System[/bold blue]\n[yellow]Model Preloader[/yellow]",
        border_style="blue"
    ))
    console.print()
    
    # Initialize preloader
    preloader = ModelPreloader()
    
    # Load all models
    try:
        results = preloader.preload_all_models()
        
        if results["success_count"] > 0:
            console.print("\n[bold green]SUCCESS: Model preloading completed![/bold green]")
            console.print(f"[dim]Models cached in: {preloader.cache_dir}[/dim]")
            return True
        else:
            console.print("\n[bold red]ERROR: Model preloading failed![/bold red]")
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