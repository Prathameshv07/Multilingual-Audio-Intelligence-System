"""
Utility Functions for Multilingual Audio Intelligence System

This module provides common helper functions, data validation utilities,
performance monitoring, and error handling functionality used across
all components of the audio intelligence system.

Key Features:
- File I/O utilities for audio and text files
- Data validation and type checking
- Performance monitoring and timing utilities
- Error handling and logging helpers
- Audio format detection and validation
- Memory management utilities
- Progress tracking for long-running operations

Dependencies: pathlib, typing, functools, time
"""

import os
import sys
import time
import logging
import functools
from pathlib import Path
from typing import Union, Optional, Dict, List, Any, Callable, Tuple
import json
import hashlib
import tempfile
import psutil
from dataclasses import dataclass
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for tracking performance metrics."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def finalize(self, end_time: float, memory_after: float, 
                success: bool = True, error_message: Optional[str] = None):
        """Finalize the metrics with end time and status."""
        self.end_time = end_time
        self.duration = end_time - self.start_time
        self.memory_after = memory_after
        self.success = success
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'duration': self.duration,
            'memory_before_mb': self.memory_before,
            'memory_after_mb': self.memory_after,
            'memory_peak_mb': self.memory_peak,
            'success': self.success,
            'error_message': self.error_message
        }


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1):
        """Update progress by increment."""
        self.current += increment
        if self.current <= self.total:
            self._display_progress()
    
    def _display_progress(self):
        """Display progress bar in console."""
        if self.total == 0:
            return
            
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f" ETA: {format_duration(eta)}"
        else:
            eta_str = ""
        
        bar_length = 30
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description}: |{bar}| {percent:.1f}% '
              f'({self.current}/{self.total}){eta_str}', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def finish(self):
        """Mark progress as complete."""
        self.current = self.total
        self._display_progress()


# File I/O Utilities
def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path: Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dict: File information including size, modification time, etc.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'exists': False}
    
    stat = file_path.stat()
    
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'is_file': file_path.is_file(),
        'is_directory': file_path.is_dir(),
        'extension': file_path.suffix.lower(),
        'name': file_path.name,
        'parent': str(file_path.parent)
    }


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        str: Hex digest of file hash
    """
    file_path = Path(file_path)
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Safe filename
    """
    # Remove or replace problematic characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = filename
    
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove multiple consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Remove leading/trailing underscores and dots
    safe_name = safe_name.strip('_.')
    
    # Ensure filename is not empty
    if not safe_name:
        safe_name = 'unnamed_file'
    
    return safe_name


# Audio File Utilities
def detect_audio_format(file_path: Union[str, Path]) -> Optional[str]:
    """
    Detect audio format from file extension and header.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Optional[str]: Detected format ('wav', 'mp3', 'ogg', 'flac', etc.)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    # Check by extension first
    extension = file_path.suffix.lower()
    extension_map = {
        '.wav': 'wav',
        '.mp3': 'mp3',
        '.ogg': 'ogg',
        '.flac': 'flac',
        '.m4a': 'm4a',
        '.aac': 'aac',
        '.wma': 'wma'
    }
    
    if extension in extension_map:
        return extension_map[extension]
    
    # Try to detect by file header
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
        
        # WAV files start with "RIFF" and contain "WAVE"
        if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return 'wav'
        
        # MP3 files often start with ID3 tag or frame sync
        if header[:3] == b'ID3' or header[:2] == b'\xFF\xFB':
            return 'mp3'
        
        # FLAC files start with "fLaC"
        if header[:4] == b'fLaC':
            return 'flac'
        
        # OGG files start with "OggS"
        if header[:4] == b'OggS':
            return 'ogg'
        
    except Exception as e:
        logger.warning(f"Failed to read file header: {e}")
    
    return None


def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate if file is a supported audio format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict: Validation results with 'valid' boolean and details
    """
    file_path = Path(file_path)
    
    result = {
        'valid': False,
        'format': None,
        'error': None,
        'file_info': get_file_info(file_path)
    }
    
    if not file_path.exists():
        result['error'] = 'File does not exist'
        return result
    
    if not file_path.is_file():
        result['error'] = 'Path is not a file'
        return result
    
    if result['file_info']['size_bytes'] == 0:
        result['error'] = 'File is empty'
        return result
    
    # Detect format
    detected_format = detect_audio_format(file_path)
    if not detected_format:
        result['error'] = 'Unsupported or unrecognized audio format'
        return result
    
    result['format'] = detected_format
    result['valid'] = True
    
    return result


# Data Validation Utilities
def validate_time_range(start_time: float, end_time: float, 
                       max_duration: Optional[float] = None) -> bool:
    """
    Validate time range parameters.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        max_duration: Optional maximum allowed duration
        
    Returns:
        bool: True if valid time range
    """
    if start_time < 0 or end_time < 0:
        return False
    
    if start_time >= end_time:
        return False
    
    if max_duration and (end_time - start_time) > max_duration:
        return False
    
    return True


def validate_language_code(lang_code: str) -> bool:
    """
    Validate ISO language code format.
    
    Args:
        lang_code: Language code to validate
        
    Returns:
        bool: True if valid format
    """
    if not isinstance(lang_code, str):
        return False
    
    lang_code = lang_code.lower().strip()
    
    # Basic validation for 2-3 character language codes
    if len(lang_code) in [2, 3] and lang_code.isalpha():
        return True
    
    return False


def validate_confidence_score(score: float) -> bool:
    """
    Validate confidence score is in valid range [0, 1].
    
    Args:
        score: Confidence score to validate
        
    Returns:
        bool: True if valid confidence score
    """
    return isinstance(score, (int, float)) and 0.0 <= score <= 1.0


# Performance Monitoring
@contextmanager
def performance_monitor(operation_name: str, 
                       log_results: bool = True) -> PerformanceMetrics:
    """
    Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation being monitored
        log_results: Whether to log results automatically
        
    Yields:
        PerformanceMetrics: Metrics object for the operation
        
    Example:
        >>> with performance_monitor("audio_processing") as metrics:
        >>>     # Your code here
        >>>     pass
        >>> print(f"Operation took {metrics.duration:.2f} seconds")
    """
    # Get initial memory usage
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    metrics = PerformanceMetrics(
        operation_name=operation_name,
        start_time=time.time(),
        memory_before=memory_before
    )
    
    try:
        yield metrics
        
        # Operation completed successfully
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        metrics.finalize(
            end_time=time.time(),
            memory_after=memory_after,
            success=True
        )
        
    except Exception as e:
        # Operation failed
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        metrics.finalize(
            end_time=time.time(),
            memory_after=memory_after,
            success=False,
            error_message=str(e)
        )
        
        if log_results:
            logger.error(f"Operation '{operation_name}' failed: {e}")
        
        raise
    
    finally:
        if log_results and metrics.duration is not None:
            if metrics.success:
                logger.info(f"Operation '{operation_name}' completed in "
                           f"{metrics.duration:.2f}s")
            
            if metrics.memory_before and metrics.memory_after:
                memory_change = metrics.memory_after - metrics.memory_before
                if abs(memory_change) > 10:  # Only log significant changes
                    logger.debug(f"Memory change: {memory_change:+.1f} MB")


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Callable: Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    
    return wrapper


# Memory Management
def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict: Memory usage in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024),
        'percent': process.memory_percent(),
        'system_total_mb': psutil.virtual_memory().total / (1024 * 1024),
        'system_available_mb': psutil.virtual_memory().available / (1024 * 1024)
    }


def check_memory_available(required_mb: float, 
                          safety_margin: float = 1.2) -> bool:
    """
    Check if sufficient memory is available.
    
    Args:
        required_mb: Required memory in MB
        safety_margin: Safety margin multiplier
        
    Returns:
        bool: True if sufficient memory available
    """
    memory = get_memory_usage()
    required_with_margin = required_mb * safety_margin
    
    return memory['system_available_mb'] >= required_with_margin


# Format Utilities
def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def truncate_text(text: str, max_length: int = 100, 
                 suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


# Error Handling Utilities
def safe_execute(func: Callable, *args, default=None, 
                log_errors: bool = True, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {e}")
        return default


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0,
                    exceptions: Tuple = (Exception,)):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for "
                                     f"{func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for "
                                   f"{func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator


# Configuration Management
def load_config(config_path: Union[str, Path], 
               default_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file with defaults.
    
    Args:
        config_path: Path to configuration file
        default_config: Default configuration values
        
    Returns:
        Dict: Configuration dictionary
    """
    config_path = Path(config_path)
    config = default_config.copy() if default_config else {}
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
    else:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
    
    return config


def save_config(config: Dict[str, Any], 
               config_path: Union[str, Path]) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        bool: True if saved successfully
    """
    config_path = Path(config_path)
    
    try:
        ensure_directory(config_path.parent)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        return False


# System Information
def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict: System information
    """
    try:
        import platform
        import torch
        
        gpu_info = "Not available"
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3) if gpu_count > 0 else 0
            gpu_info = f"{gpu_count}x {gpu_name} ({gpu_memory}GB)"
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total // (1024**3),
            'gpu_info': gpu_info,
            'torch_version': torch.__version__ if 'torch' in sys.modules else 'Not installed'
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {'error': str(e)}


# Temporary File Management
class TempFileManager:
    """Context manager for temporary file handling."""
    
    def __init__(self, prefix: str = "audio_intel_", suffix: str = ".tmp"):
        self.prefix = prefix
        self.suffix = suffix
        self.temp_files = []
    
    def create_temp_file(self, suffix: Optional[str] = None) -> str:
        """Create a temporary file and track it."""
        actual_suffix = suffix or self.suffix
        temp_file = tempfile.NamedTemporaryFile(
            prefix=self.prefix,
            suffix=actual_suffix,
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup(self):
        """Clean up all tracked temporary files."""
        for temp_path in self.temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    def main():
        """Command line interface for testing utilities."""
        parser = argparse.ArgumentParser(description="Audio Intelligence Utilities")
        parser.add_argument("--test", choices=["performance", "memory", "file", "all"],
                          default="all", help="Which utilities to test")
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose output")
        
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        if args.test in ["performance", "all"]:
            print("=== Testing Performance Monitoring ===")
            
            with performance_monitor("test_operation") as metrics:
                # Simulate some work
                time.sleep(0.1)
                data = [i**2 for i in range(1000)]
            
            print(f"Operation metrics: {metrics.to_dict()}")
            print()
        
        if args.test in ["memory", "all"]:
            print("=== Testing Memory Utilities ===")
            
            memory_info = get_memory_usage()
            print(f"Current memory usage: {memory_info}")
            
            available = check_memory_available(100)  # 100 MB
            print(f"100MB memory available: {available}")
            print()
        
        if args.test in ["file", "all"]:
            print("=== Testing File Utilities ===")
            
            # Test with a dummy file
            with TempFileManager() as temp_manager:
                temp_file = temp_manager.create_temp_file(suffix=".txt")
                
                # Write some data
                with open(temp_file, 'w') as f:
                    f.write("Test data for utilities")
                
                file_info = get_file_info(temp_file)
                print(f"File info: {file_info}")
                
                file_hash = get_file_hash(temp_file)
                print(f"File hash (MD5): {file_hash}")
                
                safe_name = safe_filename("Test <File> Name?.txt")
                print(f"Safe filename: {safe_name}")
            
            print()
        
        if args.test == "all":
            print("=== System Information ===")
            system_info = get_system_info()
            for key, value in system_info.items():
                print(f"{key}: {value}")
            print()
            
            print("=== Format Utilities ===")
            print(f"Duration format: {format_duration(3661.5)}")
            print(f"File size format: {format_file_size(1536000)}")
            print(f"Text truncation: {truncate_text('This is a very long text that should be truncated', 20)}")
    
    main() 