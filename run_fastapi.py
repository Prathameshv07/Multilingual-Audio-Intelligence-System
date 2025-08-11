#!/usr/bin/env python3
"""
Startup script for the FastAPI-based Audio Intelligence System

This script handles dependency checking, model preloading, environment setup, and application launch.
"""

import sys
import subprocess
import importlib.util
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependency(package_name, install_name=None):
    """Check if a package is installed."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install dependencies from requirements file."""
    logger.info("Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_system():
    """Check system requirements."""
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    
    # Check core dependencies
    required_packages = ['fastapi', 'uvicorn', 'jinja2', 'numpy', 'torch', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        if not check_dependency(package):
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        response = input("Install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            return install_dependencies()
        else:
            logger.error("Cannot run without required dependencies")
            return False
    
    logger.info("All dependencies are available!")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ['templates', 'static', 'uploads', 'outputs', 'model_cache']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    logger.info("Created necessary directories")

def preload_models():
    """Preload AI models before starting the server."""
    logger.info("Starting model preloading...")
    
    try:
        # Import and run model preloader
        from model_preloader import ModelPreloader
        
        preloader = ModelPreloader()
        results = preloader.preload_all_models()
        
        if results["success_count"] > 0:
            logger.info(f"✓ Model preloading completed! Loaded {results['success_count']}/{results['total_count']} models")
            return True
        else:
            logger.warning("⚠ No models loaded successfully, but continuing with application startup")
            return True  # Continue anyway for demo mode
            
    except Exception as e:
        logger.error(f"Model preloading failed: {e}")
        logger.warning("Continuing with application startup (demo mode will still work)")
        return True  # Continue anyway

def main():
    """Main startup function."""
    logger.info("Starting Audio Intelligence System (FastAPI)")
    
    # Check system requirements
    if not check_system():
        logger.error("System requirements not met")
        return 1
    
    # Create directories
    create_directories()
    
    # Check if template exists
    template_path = Path("templates/index.html")
    if not template_path.exists():
        logger.error("Template file not found: templates/index.html")
        logger.info("Please ensure the HTML template is created")
        return 1
    
    # Preload models (this is the key addition)
    preload_models()
    
    # Import and run the FastAPI app
    try:
        logger.info("Starting FastAPI server...")
        logger.info("Access the application at: http://127.0.0.1:8000")
        logger.info("API documentation at: http://127.0.0.1:8000/api/docs")
        
        # Import uvicorn here to avoid import errors during dependency check
        import uvicorn
        
        # Run the server
        uvicorn.run(
            "web_app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 