#!/usr/bin/env python3
"""
Consolidated Audio Intelligence System Runner

This script provides a unified way to run the system with different modes:
- Web App Mode: Interactive web interface
- Demo Mode: Test system capabilities
- CLI Mode: Command-line processing
- Test Mode: System validation

Usage:
    python run_app.py [--mode web|demo|cli|test] [--port PORT] [--host HOST]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_web_app(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the web application."""
    logger.info("🌐 Starting Web Application...")
    
    try:
        # Use the working web_app.py directly
        import uvicorn
        from web_app import app
        
        uvicorn.run(app, host=host, port=port, log_level="info" if debug else "warning")
        
    except Exception as e:
        logger.error(f"❌ Failed to start web app: {e}")
        sys.exit(1)

def run_demo():
    """Run the demo system."""
    logger.info("🎵 Starting Demo System...")
    
    try:
        from src.demo import main
        main()
        
    except Exception as e:
        logger.error(f"❌ Failed to run demo: {e}")
        sys.exit(1)

def run_tests():
    """Run system tests."""
    logger.info("🧪 Running System Tests...")
    
    try:
        from src.test_system import main
        main()
        
    except Exception as e:
        logger.error(f"❌ Failed to run tests: {e}")
        sys.exit(1)

def run_cli_mode():
    """Run CLI processing mode."""
    logger.info("💻 Starting CLI Mode...")
    
    try:
        from src.main import main
        main()
        
    except Exception as e:
        logger.error(f"❌ Failed to start CLI mode: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    required_modules = [
        'src.translator',
        'src.audio_processor',
        'src.main', 
        'web_app'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError as e:
            logger.error(f"❌ {module}: {e}")
            missing.append(module)
    
    if missing:
        logger.error(f"❌ Missing modules: {', '.join(missing)}")
        logger.error("Install dependencies with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies available")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audio Intelligence System Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run web app (default)
  python run_app.py --mode demo       # Run demo system
  python run_app.py --mode test       # Run system tests
  python run_app.py --mode cli        # Run CLI mode
  python run_app.py --port 8080       # Run web app on port 8080
  python run_app.py --host localhost  # Run web app on localhost only
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "demo", "cli", "test"], 
        default="web",
        help="Run mode (default: web)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port for web app (default: 8000)"
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host for web app (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--skip-deps", 
        action="store_true",
        help="Skip dependency checking"
    )
    
    args = parser.parse_args()
    
    logger.info("🎵 Audio Intelligence System")
    logger.info("=" * 50)
    
    # Check dependencies unless skipped
    if not args.skip_deps:
        if not check_dependencies():
            logger.error("❌ Critical dependencies missing. Exiting.")
            sys.exit(1)
    
    # Run selected mode
    if args.mode == "web":
        run_web_app(host=args.host, port=args.port, debug=args.debug)
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "cli":
        run_cli_mode()
    else:
        logger.error(f"❌ Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
