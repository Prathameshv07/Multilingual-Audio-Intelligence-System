"""
Audio Preprocessing Module for Multilingual Audio Intelligence System

This module handles the standardization of diverse audio inputs into a consistent
format suitable for downstream ML models. It supports various audio formats
(wav, mp3, ogg, flac), sample rates (8k-48k), bit depths (4-32 bits), and 
handles SNR variations as specified in PS-6 requirements.

Key Features:
- Format conversion and standardization
- Intelligent resampling to 16kHz
- Stereo to mono conversion
- Volume normalization for SNR robustness
- Memory-efficient processing
- Robust error handling

Dependencies: pydub, librosa, numpy
System Dependencies: ffmpeg (for format conversion)
"""

import os
import logging
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.utils import which
from typing import Tuple, Optional, Union
import tempfile
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class AudioProcessor:
    """
    Handles audio preprocessing for the multilingual audio intelligence system.
    
    This class standardizes diverse audio inputs into a consistent format:
    - 16kHz sample rate (optimal for ASR models)
    - Single channel (mono)
    - Float32 numpy array format
    - Normalized amplitude
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize AudioProcessor with target specifications.
        
        Args:
            target_sample_rate (int): Target sample rate in Hz. Default 16kHz
                                    optimized for Whisper and pyannote models.
        """
        self.target_sample_rate = target_sample_rate
        self.supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']
        
        # Verify ffmpeg availability
        if not which("ffmpeg"):
            logger.warning("ffmpeg not found. Some format conversions may fail.")
    
    def process_audio(self, audio_input: Union[str, bytes, np.ndarray], 
                     input_sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Main processing function that standardizes any audio input.
        
        Args:
            audio_input: Can be file path (str), audio bytes, or numpy array
            input_sample_rate: Required if audio_input is numpy array
            
        Returns:
            Tuple[np.ndarray, int]: (processed_audio_array, sample_rate)
            
        Raises:
            ValueError: If input format is unsupported or invalid
            FileNotFoundError: If audio file doesn't exist
            Exception: For processing errors
        """
        try:
            # Determine input type and load audio
            if isinstance(audio_input, str):
                # File path input
                audio_array, original_sr = self._load_from_file(audio_input)
            elif isinstance(audio_input, bytes):
                # Bytes input (e.g., from uploaded file)
                audio_array, original_sr = self._load_from_bytes(audio_input)
            elif isinstance(audio_input, np.ndarray):
                # Numpy array input
                if input_sample_rate is None:
                    raise ValueError("input_sample_rate must be provided for numpy array input")
                audio_array = audio_input.astype(np.float32)
                original_sr = input_sample_rate
            else:
                raise ValueError(f"Unsupported input type: {type(audio_input)}")
            
            logger.info(f"Loaded audio: {audio_array.shape}, {original_sr}Hz")
            
            # Apply preprocessing pipeline
            processed_audio = self._preprocess_pipeline(audio_array, original_sr)
            
            logger.info(f"Processed audio: {processed_audio.shape}, {self.target_sample_rate}Hz")
            
            return processed_audio, self.target_sample_rate
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            raise
    
    def _load_from_file(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format {file_ext}. Supported: {self.supported_formats}")
        
        try:
            # Use librosa for robust loading with automatic resampling
            audio_array, sample_rate = librosa.load(file_path, sr=None, mono=False)
            return audio_array, sample_rate
        except Exception as e:
            # Fallback to pydub for format conversion
            logger.warning(f"librosa failed, trying pydub: {e}")
            return self._load_with_pydub(file_path)
    
    def _load_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes (e.g., uploaded file)."""
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Try to detect format and load
            audio_array, sample_rate = self._load_with_pydub(tmp_path)
            return audio_array, sample_rate
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    
    def _load_with_pydub(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio using pydub with format detection."""
        try:
            # Let pydub auto-detect format
            audio_segment = AudioSegment.from_file(file_path)
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Handle stereo audio
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Normalize to [-1, 1] range
            samples = samples / (2**15)  # 16-bit normalization
            
            return samples, audio_segment.frame_rate
            
        except Exception as e:
            raise Exception(f"Failed to load audio with pydub: {str(e)}")
    
    def _preprocess_pipeline(self, audio_array: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Apply the complete preprocessing pipeline.
        
        Pipeline steps:
        1. Convert stereo to mono
        2. Resample to target sample rate
        3. Normalize amplitude
        4. Apply basic noise reduction (optional)
        """
        # Step 1: Convert to mono if stereo
        if len(audio_array.shape) > 1 and audio_array.shape[0] == 2:
            # librosa format: (channels, samples) for stereo
            audio_array = np.mean(audio_array, axis=0)
        elif len(audio_array.shape) > 1 and audio_array.shape[1] == 2:
            # pydub format: (samples, channels) for stereo
            audio_array = np.mean(audio_array, axis=1)
        
        # Ensure 1D array
        audio_array = audio_array.flatten()
        
        logger.debug(f"After mono conversion: {audio_array.shape}")
        
        # Step 2: Resample if necessary
        if original_sr != self.target_sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=original_sr, 
                target_sr=self.target_sample_rate,
                res_type='kaiser_best'  # High quality resampling
            )
            logger.debug(f"Resampled from {original_sr}Hz to {self.target_sample_rate}Hz")
        
        # Step 3: Amplitude normalization
        audio_array = self._normalize_audio(audio_array)
        
        # Step 4: Basic preprocessing for robustness
        audio_array = self._apply_preprocessing_filters(audio_array)
        
        return audio_array.astype(np.float32)
    
    def _normalize_audio(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude to handle varying SNR conditions.
        
        Uses RMS-based normalization for better handling of varying
        signal-to-noise ratios (-5dB to 20dB as per PS-6 requirements).
        """
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio_array**2))
        
        if rms > 0:
            # Target RMS level (prevents over-amplification)
            target_rms = 0.1
            normalization_factor = target_rms / rms
            
            # Apply normalization with clipping protection
            normalized = audio_array * normalization_factor
            normalized = np.clip(normalized, -1.0, 1.0)
            
            logger.debug(f"RMS normalization: {rms:.4f} -> {target_rms:.4f}")
            return normalized
        
        return audio_array
    
    def _apply_preprocessing_filters(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing filters for improved robustness.
        
        Includes:
        - DC offset removal
        - Light high-pass filtering (removes very low frequencies)
        """
        # Remove DC offset
        audio_array = audio_array - np.mean(audio_array)
        
        # Simple high-pass filter to remove very low frequencies (< 80Hz)
        # This helps with handling background noise and rumble
        try:
            from scipy.signal import butter, filtfilt
            
            # Design high-pass filter
            nyquist = self.target_sample_rate / 2
            cutoff = 80 / nyquist  # 80Hz cutoff
            
            if cutoff < 1.0:  # Valid frequency range
                b, a = butter(N=1, Wn=cutoff, btype='high')
                audio_array = filtfilt(b, a, audio_array)
                logger.debug("Applied high-pass filter (80Hz cutoff)")
                
        except ImportError:
            logger.debug("scipy not available, skipping high-pass filter")
        except Exception as e:
            logger.debug(f"High-pass filter failed: {e}")
        
        return audio_array
    
    def get_audio_info(self, audio_input: Union[str, bytes]) -> dict:
        """
        Get detailed information about audio file without full processing.
        
        Returns:
            dict: Audio metadata including duration, sample rate, channels, etc.
        """
        try:
            if isinstance(audio_input, str):
                # File path
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                audio_segment = AudioSegment.from_file(audio_input)
            else:
                # Bytes input
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(audio_input)
                    tmp_path = tmp_file.name
                
                try:
                    audio_segment = AudioSegment.from_file(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            
            return {
                'duration_seconds': len(audio_segment) / 1000.0,
                'sample_rate': audio_segment.frame_rate,
                'channels': audio_segment.channels,
                'sample_width': audio_segment.sample_width,
                'frame_count': audio_segment.frame_count(),
                'max_possible_amplitude': audio_segment.max_possible_amplitude
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}


# Utility functions for common audio operations
def validate_audio_file(file_path: str) -> bool:
    """
    Quick validation of audio file without full loading.
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        bool: True if file appears to be valid audio
    """
    try:
        processor = AudioProcessor()
        info = processor.get_audio_info(file_path)
        return info.get('duration_seconds', 0) > 0
    except Exception:
        return False


def estimate_processing_time(file_path: str) -> float:
    """
    Estimate processing time based on audio duration.
    
    Args:
        file_path (str): Path to audio file
        
    Returns:
        float: Estimated processing time in seconds
    """
    try:
        processor = AudioProcessor()
        info = processor.get_audio_info(file_path)
        duration = info.get('duration_seconds', 0)
        
        # Rough estimate: 0.1x to 0.3x real-time for preprocessing
        # depending on format conversion needs
        estimated_time = duration * 0.2
        return max(estimated_time, 1.0)  # Minimum 1 second
    except Exception:
        return 10.0  # Default estimate


if __name__ == "__main__":
    # Example usage and testing
    processor = AudioProcessor()
    
    # Test with a sample file (if available)
    test_files = ["sample.wav", "sample.mp3", "test_audio.flac"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                print(f"\nTesting {test_file}:")
                
                # Get info
                info = processor.get_audio_info(test_file)
                print(f"Info: {info}")
                
                # Process
                audio, sr = processor.process_audio(test_file)
                print(f"Processed: shape={audio.shape}, sr={sr}")
                
                # Validate
                is_valid = validate_audio_file(test_file)
                print(f"Valid: {is_valid}")
                
            except Exception as e:
                print(f"Error processing {test_file}: {e}")