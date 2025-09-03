#!/usr/bin/env python3
"""
Startup script for Hugging Face Spaces deployment.
"""

import os
import subprocess
import time
import sys

def main():
    """Main startup function."""
    print('🚀 Starting Multilingual Audio Intelligence System...')
    
    # Create necessary directories
    dirs = [
        'uploads', 'outputs', 'model_cache', 'temp_files', 
        'demo_results', '/tmp/matplotlib', '/tmp/fontconfig'
    ]
    
    for d in dirs:
        try:
            os.makedirs(d, mode=0o777, exist_ok=True)
            print(f'✅ Created directory: {d}')
        except Exception as e:
            print(f'⚠️ Failed to create directory {d}: {e}')
    
    # Try to preload models
    try:
        print('🔄 Preloading models...')
        result = subprocess.run(['python', 'model_preloader.py'], 
                              check=True, capture_output=True, text=True)
        print('✅ Models loaded successfully')
        if result.stdout:
            print(f'Model preloader output: {result.stdout}')
    except subprocess.CalledProcessError as e:
        print(f'⚠️ Model preloading failed: {e}')
        if e.stdout:
            print(f'Model preloader stdout: {e.stdout}')
        if e.stderr:
            print(f'Model preloader stderr: {e.stderr}')
        print('🔄 Continuing without preloaded models...')
    except Exception as e:
        print(f'⚠️ Model preloading failed: {e}')
        print('🔄 Continuing without preloaded models...')
    
    # Start the web application
    print('🌐 Starting web application...')
    try:
        import uvicorn
        uvicorn.run('web_app:app', host='0.0.0.0', port=7860, workers=1, log_level='info')
    except Exception as e:
        print(f'❌ Failed to start web application: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
