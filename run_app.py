#!/usr/bin/env python3
"""
Simple startup script for local development.
This is the main entry point for running the application locally.
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Main function for local development."""
    print("🚀 Starting Audio Intelligence System (Local Development)")
    
    # Create necessary directories
    dirs = ['uploads', 'outputs', 'model_cache', 'temp_files', 'demo_results']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        print(f"✅ Created directory: {d}")
    
    # Check if web_app.py exists
    if not Path("web_app.py").exists():
        print("❌ web_app.py not found!")
        sys.exit(1)
    
    print("🌐 Starting web application...")
    print("📍 Access the application at: http://127.0.0.1:8000")
    print("📚 API documentation at: http://127.0.0.1:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Start the server with reload for development
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()