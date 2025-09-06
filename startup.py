#!/usr/bin/env python3
"""
Startup script for Hugging Face Spaces deployment.
Handles model preloading and graceful fallbacks for containerized environments.
"""

# Suppress ONNX Runtime warnings BEFORE any imports
# import warnings
# warnings.filterwarnings("ignore", message=".*executable stack.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

import os
import subprocess
import sys
import logging

# # Set critical environment variables immediately
# os.environ.update({
#     'ORT_DYLIB_DEFAULT_OPTIONS': 'DisableExecutablePageAllocator=1',
#     'ONNXRUNTIME_EXECUTION_PROVIDERS': 'CPUExecutionProvider',
#     'ORT_DISABLE_TLS_ARENA': '1'
# })

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories with proper permissions."""
    dirs = [
        'uploads', 'outputs', 'model_cache', 'temp_files', 
        'demo_results', '/tmp/matplotlib', '/tmp/fontconfig'
    ]
    
    for d in dirs:
        try:
            os.makedirs(d, mode=0o777, exist_ok=True)
            logger.info(f'✅ Created directory: {d}')
        except Exception as e:
            logger.warning(f'⚠️ Failed to create directory {d}: {e}')

def check_dependencies():
    """Check if critical dependencies are available."""
    critical_deps = ['fastapi', 'uvicorn', 'torch', 'transformers']
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
            logger.info(f'✅ {dep} available')
        except ImportError:
            missing_deps.append(dep)
            logger.warning(f'⚠️ {dep} not available')
    
    if missing_deps:
        logger.error(f'❌ Missing critical dependencies: {missing_deps}')
        return False
    
    return True

def preload_models():
    """Attempt to preload models with graceful fallback."""
    logger.info('🔄 Attempting model preloading...')
    
    try:
        # Check if model_preloader.py exists and is importable
        import model_preloader
        logger.info('✅ Model preloader module found')
        
        # Set comprehensive environment variables for ONNX Runtime
        # env = os.environ.copy()
        # env.update({
        #     'ORT_DYLIB_DEFAULT_OPTIONS': 'DisableExecutablePageAllocator=1',
        #     'ONNXRUNTIME_EXECUTION_PROVIDERS': 'CPUExecutionProvider',
        #     'ORT_DISABLE_TLS_ARENA': '1',
        #     'TF_ENABLE_ONEDNN_OPTS': '0',
        #     'OMP_NUM_THREADS': '1',
        #     'MKL_NUM_THREADS': '1',
        #     'NUMBA_NUM_THREADS': '1',
        #     'TOKENIZERS_PARALLELISM': 'false',
        #     'MALLOC_ARENA_MAX': '2',
        #     # Additional ONNX Runtime fixes
        #     'ONNXRUNTIME_LOG_SEVERITY_LEVEL': '3',
        #     'ORT_LOGGING_LEVEL': 'WARNING'
        # })
        
        # # Try to fix ONNX Runtime libraries before running preloader
        # try:
        #     import subprocess
        #     subprocess.run([
        #         'find', '/usr/local/lib/python*/site-packages/onnxruntime', 
        #         '-name', '*.so', '-exec', 'execstack', '-c', '{}', ';'
        #     ], capture_output=True, timeout=30)
        # except:
        #     pass  # Continue if execstack fix fails
        
        # Try to run the preloader
        result = subprocess.run(
            ['python', 'model_preloader.py'], 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
            # env=env
        )
        
        if result.returncode == 0:
            logger.info('✅ Models loaded successfully')
            if result.stdout:
                logger.info(f'Preloader output: {result.stdout[:500]}...')
            return True
        else:
            logger.warning(f'⚠️ Model preloading failed with return code {result.returncode}')
            # if result.stderr:
            #     # Filter out expected ONNX warnings
            #     stderr_lines = result.stderr.split('\n')
            #     important_errors = [line for line in stderr_lines 
            #                       if 'executable stack' not in line.lower() 
            #                       and 'onnxruntime' not in line.lower() 
            #                       and line.strip()]
            #     if important_errors:
            #         logger.warning(f'Important errors: {important_errors[:3]}')
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning('⚠️ Model preloading timed out, continuing...')
        return False
    except Exception as e:
        if 'executable stack' not in str(e).lower():
            logger.warning(f'⚠️ Model preloading failed: {e}')
        return False

def start_web_app():
    """Start the web application."""
    logger.info('🌐 Starting web application...')
    
    try:
        import uvicorn
        logger.info('✅ Uvicorn imported successfully')
        
        # Start the server
        uvicorn.run(
            'web_app:app', 
            host='0.0.0.0', 
            port=7860, 
            workers=1, 
            log_level='info',
            access_log=True
        )
    except ImportError as e:
        logger.error(f'❌ Failed to import uvicorn: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'❌ Failed to start web application: {e}')
        sys.exit(1)

def main():
    """Main startup function."""
    logger.info('🚀 Starting Multilingual Audio Intelligence System...')
    
    # Check critical dependencies
    if not check_dependencies():
        logger.error('❌ Critical dependencies missing, exiting...')
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Attempt model preloading (non-blocking)
    preload_models()
    
    # Start the web application
    start_web_app()

if __name__ == '__main__':
    main()
