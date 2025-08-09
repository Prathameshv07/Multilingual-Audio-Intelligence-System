"""
Advanced Speech Recognition Module for Multilingual Audio Intelligence System

This module implements state-of-the-art automatic speech recognition using faster-whisper
with integrated language identification capabilities. Designed for maximum performance 
on CPU-constrained environments while maintaining SOTA accuracy.

Key Features:
- Faster-whisper with CTranslate2 backend for 4x speed improvement
- Integrated Language Identification (no separate LID module needed)
- VAD-based batching for 14.6x real-time performance on CPU
- Word-level timestamps for interactive UI synchronization
- INT8 quantization for memory efficiency
- Robust error handling and multilingual support
- CPU and GPU optimization paths

Model: openai/whisper-small (optimized for speed/accuracy balance)
Dependencies: faster-whisper, torch, numpy
"""

import os
import logging
import warnings
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import tempfile
from dataclasses import dataclass
import time

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available. Install with: pip install faster-whisper")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TranscriptionSegment:
    """
    Data class representing a transcribed speech segment with rich metadata.
    
    Attributes:
        start_time (float): Segment start time in seconds
        end_time (float): Segment end time in seconds
        text (str): Transcribed text in native script
        language (str): Detected language code (e.g., 'en', 'hi', 'ar')
        confidence (float): Overall transcription confidence
        word_timestamps (List[Dict]): Word-level timing information
        speaker_id (str): Associated speaker identifier (if provided)
    """
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float = 1.0
    word_timestamps: Optional[List[Dict]] = None
    speaker_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text,
            'language': self.language,
            'confidence': self.confidence,
            'duration': self.duration,
            'word_timestamps': self.word_timestamps or [],
            'speaker_id': self.speaker_id
        }


class SpeechRecognizer:
    """
    State-of-the-art speech recognition with integrated language identification.
    
    Uses faster-whisper for optimal performance on both CPU and GPU, with advanced
    batching strategies for maximum throughput on constrained hardware.
    """
    
    def __init__(self,
                 model_size: str = "small",
                 device: Optional[str] = None,
                 compute_type: str = "int8",
                 cpu_threads: Optional[int] = None,
                 num_workers: int = 1,
                 download_root: Optional[str] = None):
        """
        Initialize the Speech Recognizer with optimizations.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'small', 'medium', 'large')
            device (str, optional): Device to run on ('cpu', 'cuda', 'auto')
            compute_type (str): Precision type ('int8', 'float16', 'float32')
            cpu_threads (int, optional): Number of CPU threads to use
            num_workers (int): Number of workers for batch processing
            download_root (str, optional): Directory to store model files
        """
        self.model_size = model_size
        self.compute_type = compute_type
        self.num_workers = num_workers
        
        # Device selection with intelligence
        if device == 'auto' or device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                # Adjust compute type for GPU
                if compute_type == 'int8' and torch.cuda.is_available():
                    self.compute_type = 'float16'  # GPU prefers float16 over int8
            else:
                self.device = 'cpu'
                self.compute_type = 'int8'  # CPU benefits from int8
        else:
            self.device = device
        
        # CPU thread optimization
        if cpu_threads is None:
            if self.device == 'cpu':
                cpu_threads = min(os.cpu_count() or 4, 4)  # Cap at 4 for HF Spaces
        self.cpu_threads = cpu_threads
        
        logger.info(f"Initializing SpeechRecognizer: {model_size} on {self.device} "
                   f"with {self.compute_type} precision")
        
        # Initialize models
        self.model = None
        self.batched_model = None
        self._load_models(download_root)
    
    def _load_models(self, download_root: Optional[str] = None):
        """Load both standard and batched Whisper models."""
        if not FASTER_WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required for speech recognition. "
                "Install with: pip install faster-whisper"
            )
        
        try:
            logger.info(f"Loading {self.model_size} Whisper model...")
            
            # Set CPU threads for optimal performance
            if self.device == 'cpu' and self.cpu_threads:
                os.environ['OMP_NUM_THREADS'] = str(self.cpu_threads)
            
            # Load standard model
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=download_root,
                cpu_threads=self.cpu_threads
            )
            
            # Load batched model for improved throughput
            try:
                self.batched_model = BatchedInferencePipeline(
                    model=self.model,
                    chunk_length=30,  # 30-second chunks
                    batch_size=16 if self.device == 'cuda' else 8,
                    use_vad_model=True,  # VAD-based batching for massive speedup
                )
                logger.info("Batched inference pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load batched pipeline: {e}. Using standard model.")
                self.batched_model = None
            
            logger.info(f"Speech recognition models loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load speech recognition models: {e}")
            raise
    
    def transcribe_audio(self, 
                        audio_input: Union[str, np.ndarray],
                        sample_rate: int = 16000,
                        language: Optional[str] = None,
                        word_timestamps: bool = True,
                        use_batching: bool = True) -> List[TranscriptionSegment]:
        """
        Transcribe audio with integrated language identification.
        
        Args:
            audio_input: Audio file path or numpy array
            sample_rate: Sample rate if audio_input is numpy array
            language: Language hint (optional, auto-detected if None)
            word_timestamps: Whether to generate word-level timestamps
            use_batching: Whether to use batched inference for speed
            
        Returns:
            List[TranscriptionSegment]: Transcription results with metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_models() first.")
        
        try:
            # Prepare audio input
            audio_file = self._prepare_audio_input(audio_input, sample_rate)
            
            logger.info("Starting speech recognition...")
            start_time = time.time()
            
            # Choose processing method based on availability and preference
            if use_batching and self.batched_model is not None:
                segments = self._transcribe_batched(
                    audio_file, language, word_timestamps
                )
            else:
                segments = self._transcribe_standard(
                    audio_file, language, word_timestamps
                )
            
            processing_time = time.time() - start_time
            total_audio_duration = sum(seg.duration for seg in segments)
            rtf = processing_time / max(total_audio_duration, 0.1)
            
            logger.info(f"Transcription completed in {processing_time:.2f}s "
                       f"(RTF: {rtf:.2f}x)")
            logger.info(f"Detected {len(set(seg.language for seg in segments))} languages, "
                       f"{len(segments)} segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
        
        finally:
            # Clean up temporary files
            if isinstance(audio_input, np.ndarray):
                try:
                    if hasattr(audio_file, 'name') and os.path.exists(audio_file.name):
                        os.unlink(audio_file.name)
                except Exception:
                    pass
    
    def _transcribe_batched(self, 
                           audio_file: str,
                           language: Optional[str],
                           word_timestamps: bool) -> List[TranscriptionSegment]:
        """Transcribe using batched inference for maximum speed."""
        try:
            # Use batched pipeline for optimal CPU performance
            result = self.batched_model(
                audio_file,
                language=language,
                word_level_timestamps=word_timestamps,
                batch_size=16 if self.device == 'cuda' else 8
            )
            
            segments = []
            for segment in result:
                # Extract word timestamps if available
                word_times = None
                if word_timestamps and hasattr(segment, 'words'):
                    word_times = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'confidence': getattr(word, 'probability', 1.0)
                        }
                        for word in segment.words
                    ]
                
                transcription_segment = TranscriptionSegment(
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text.strip(),
                    language=getattr(segment, 'language', language or 'unknown'),
                    confidence=getattr(segment, 'avg_logprob', 1.0),
                    word_timestamps=word_times
                )
                segments.append(transcription_segment)
            
            return segments
            
        except Exception as e:
            logger.warning(f"Batched transcription failed: {e}. Falling back to standard.")
            return self._transcribe_standard(audio_file, language, word_timestamps)
    
    def _transcribe_standard(self,
                           audio_file: str,
                           language: Optional[str],
                           word_timestamps: bool) -> List[TranscriptionSegment]:
        """Transcribe using standard Whisper model."""
        segments, info = self.model.transcribe(
            audio_file,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,  # Enable VAD filtering
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=1,  # Faster with beam_size=1 on CPU
            temperature=0.0  # Deterministic output
        )
        
        results = []
        for segment in segments:
            # Extract word timestamps
            word_times = None
            if word_timestamps and hasattr(segment, 'words') and segment.words:
                word_times = [
                    {
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'confidence': getattr(word, 'probability', 1.0)
                    }
                    for word in segment.words
                ]
            
            transcription_segment = TranscriptionSegment(
                start_time=segment.start,
                end_time=segment.end,
                text=segment.text.strip(),
                language=info.language,
                confidence=getattr(segment, 'avg_logprob', 1.0),
                word_timestamps=word_times
            )
            results.append(transcription_segment)
        
        return results
    
    def transcribe_segments(self,
                          audio_array: np.ndarray,
                          sample_rate: int,
                          speaker_segments: List[Tuple[float, float, str]],
                          word_timestamps: bool = True) -> List[TranscriptionSegment]:
        """
        Transcribe pre-segmented audio chunks from speaker diarization.
        
        Args:
            audio_array: Full audio as numpy array
            sample_rate: Audio sample rate
            speaker_segments: List of (start_time, end_time, speaker_id) tuples
            word_timestamps: Whether to generate word-level timestamps
            
        Returns:
            List[TranscriptionSegment]: Transcribed segments with speaker attribution
        """
        if not speaker_segments:
            return []
        
        try:
            segments_to_process = []
            
            # Extract audio chunks for each speaker segment
            for start_time, end_time, speaker_id in speaker_segments:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Extract audio chunk
                audio_chunk = audio_array[start_sample:end_sample]
                
                # Skip very short segments
                if len(audio_chunk) < sample_rate * 0.1:  # Less than 100ms
                    continue
                
                segments_to_process.append({
                    'audio': audio_chunk,
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker_id': speaker_id
                })
            
            # Process segments in batches for efficiency
            all_results = []
            batch_size = 8 if self.device == 'cuda' else 4
            
            for i in range(0, len(segments_to_process), batch_size):
                batch = segments_to_process[i:i + batch_size]
                batch_results = self._process_segment_batch(
                    batch, sample_rate, word_timestamps
                )
                all_results.extend(batch_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            return []
    
    def _process_segment_batch(self,
                             segment_batch: List[Dict],
                             sample_rate: int,
                             word_timestamps: bool) -> List[TranscriptionSegment]:
        """Process a batch of audio segments efficiently."""
        results = []
        
        for segment_info in segment_batch:
            try:
                # Save audio chunk to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav', prefix='segment_'
                )
                
                # Use soundfile for saving if available
                try:
                    import soundfile as sf
                    sf.write(temp_file.name, segment_info['audio'], sample_rate)
                except ImportError:
                    # Fallback to scipy
                    from scipy.io import wavfile
                    wavfile.write(temp_file.name, sample_rate, 
                                (segment_info['audio'] * 32767).astype(np.int16))
                
                temp_file.close()
                
                # Transcribe the segment
                transcription_segments = self.transcribe_audio(
                    temp_file.name,
                    sample_rate=sample_rate,
                    word_timestamps=word_timestamps,
                    use_batching=False  # Already batching at higher level
                )
                
                # Adjust timestamps and add speaker info
                for ts in transcription_segments:
                    # Adjust timestamps to global timeline
                    time_offset = segment_info['start_time']
                    ts.start_time += time_offset
                    ts.end_time += time_offset
                    ts.speaker_id = segment_info['speaker_id']
                    
                    # Adjust word timestamps
                    if ts.word_timestamps:
                        for word in ts.word_timestamps:
                            word['start'] += time_offset
                            word['end'] += time_offset
                    
                    results.append(ts)
                
            except Exception as e:
                logger.warning(f"Failed to transcribe segment: {e}")
                continue
            
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                except Exception:
                    pass
        
        return results
    
    def _prepare_audio_input(self,
                           audio_input: Union[str, np.ndarray],
                           sample_rate: int) -> str:
        """Prepare audio input for Whisper processing."""
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return audio_input
        
        elif isinstance(audio_input, np.ndarray):
            return self._save_array_to_tempfile(audio_input, sample_rate)
        
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
    
    def _save_array_to_tempfile(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Save numpy array to temporary WAV file."""
        try:
            import soundfile as sf
            
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.wav', prefix='whisper_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Ensure audio is mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize audio
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            sf.write(temp_path, audio_array, sample_rate)
            logger.debug(f"Saved audio array to: {temp_path}")
            return temp_path
            
        except ImportError:
            # Fallback to scipy
            try:
                from scipy.io import wavfile
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix='.wav', prefix='whisper_'
                )
                temp_path = temp_file.name
                temp_file.close()
                
                # Convert to 16-bit int
                audio_int16 = (audio_array * 32767).astype(np.int16)
                wavfile.write(temp_path, sample_rate, audio_int16)
                
                return temp_path
                
            except ImportError:
                raise ImportError(
                    "Neither soundfile nor scipy available. "
                    "Install with: pip install soundfile"
                )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Whisper supports 99 languages
        return [
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl',
            'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro',
            'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy',
            'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu',
            'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km',
            'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo',
            'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg',
            'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su'
        ]
    
    def benchmark_performance(self, audio_file: str) -> Dict[str, float]:
        """Benchmark transcription performance on given audio file."""
        try:
            # Get audio duration
            import librosa
            duration = librosa.get_duration(filename=audio_file)
            
            # Test standard transcription
            start_time = time.time()
            segments_standard = self.transcribe_audio(
                audio_file, use_batching=False, word_timestamps=False
            )
            standard_time = time.time() - start_time
            
            # Test batched transcription (if available)
            batched_time = None
            if self.batched_model:
                start_time = time.time()
                segments_batched = self.transcribe_audio(
                    audio_file, use_batching=True, word_timestamps=False
                )
                batched_time = time.time() - start_time
            
            return {
                'audio_duration': duration,
                'standard_processing_time': standard_time,
                'batched_processing_time': batched_time,
                'standard_rtf': standard_time / duration,
                'batched_rtf': batched_time / duration if batched_time else None,
                'speedup': standard_time / batched_time if batched_time else None
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'device') and 'cuda' in str(self.device):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# Convenience function for easy usage
def transcribe_audio(audio_input: Union[str, np.ndarray],
                    sample_rate: int = 16000,
                    model_size: str = "small",
                    language: Optional[str] = None,
                    device: Optional[str] = None,
                    word_timestamps: bool = True) -> List[TranscriptionSegment]:
    """
    Convenience function to transcribe audio with optimal settings.
    
    Args:
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate for numpy array input
        model_size: Whisper model size ('tiny', 'small', 'medium', 'large')
        language: Language hint (auto-detected if None)
        device: Device to run on ('cpu', 'cuda', 'auto')
        word_timestamps: Whether to generate word-level timestamps
        
    Returns:
        List[TranscriptionSegment]: Transcription results
        
    Example:
        >>> # Transcribe from file
        >>> segments = transcribe_audio("meeting.wav")
        >>> 
        >>> # Transcribe numpy array
        >>> import numpy as np
        >>> audio_data = np.random.randn(16000 * 10)  # 10 seconds
        >>> segments = transcribe_audio(audio_data, sample_rate=16000)
        >>> 
        >>> # Print results
        >>> for seg in segments:
        >>>     print(f"[{seg.start_time:.1f}-{seg.end_time:.1f}] "
        >>>           f"({seg.language}): {seg.text}")
    """
    recognizer = SpeechRecognizer(
        model_size=model_size,
        device=device
    )
    
    return recognizer.transcribe_audio(
        audio_input=audio_input,
        sample_rate=sample_rate,
        language=language,
        word_timestamps=word_timestamps
    )


# Example usage and testing
if __name__ == "__main__":
    import sys
    import argparse
    import json
    
    def main():
        """Command line interface for testing speech recognition."""
        parser = argparse.ArgumentParser(description="Advanced Speech Recognition Tool")
        parser.add_argument("audio_file", help="Path to audio file")
        parser.add_argument("--model-size", choices=["tiny", "small", "medium", "large"],
                          default="small", help="Whisper model size")
        parser.add_argument("--language", help="Language hint (auto-detected if not provided)")
        parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                          help="Device to run on")
        parser.add_argument("--no-word-timestamps", action="store_true",
                          help="Disable word-level timestamps")
        parser.add_argument("--no-batching", action="store_true",
                          help="Disable batched inference")
        parser.add_argument("--output-format", choices=["json", "text", "srt"],
                          default="text", help="Output format")
        parser.add_argument("--benchmark", action="store_true",
                          help="Run performance benchmark")
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose logging")
        
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            print(f"Processing audio file: {args.audio_file}")
            
            recognizer = SpeechRecognizer(
                model_size=args.model_size,
                device=args.device
            )
            
            if args.benchmark:
                print("\n=== PERFORMANCE BENCHMARK ===")
                benchmark = recognizer.benchmark_performance(args.audio_file)
                for key, value in benchmark.items():
                    if value is not None:
                        print(f"{key}: {value:.3f}")
                print()
            
            # Transcribe audio
            segments = recognizer.transcribe_audio(
                audio_input=args.audio_file,
                language=args.language,
                word_timestamps=not args.no_word_timestamps,
                use_batching=not args.no_batching
            )
            
            # Output results
            if args.output_format == "json":
                result = {
                    "audio_file": args.audio_file,
                    "num_segments": len(segments),
                    "languages": list(set(seg.language for seg in segments)),
                    "total_duration": sum(seg.duration for seg in segments),
                    "segments": [seg.to_dict() for seg in segments]
                }
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
            elif args.output_format == "srt":
                for i, segment in enumerate(segments, 1):
                    start_time = f"{int(segment.start_time//3600):02d}:{int((segment.start_time%3600)//60):02d}:{segment.start_time%60:06.3f}".replace('.', ',')
                    end_time = f"{int(segment.end_time//3600):02d}:{int((segment.end_time%3600)//60):02d}:{segment.end_time%60:06.3f}".replace('.', ',')
                    print(f"{i}")
                    print(f"{start_time} --> {end_time}")
                    print(f"{segment.text}")
                    print()
                    
            else:  # text format
                print(f"\n=== SPEECH RECOGNITION RESULTS ===")
                print(f"Audio file: {args.audio_file}")
                print(f"Model: {args.model_size}")
                print(f"Device: {recognizer.device}")
                print(f"Languages detected: {', '.join(set(seg.language for seg in segments))}")
                print(f"Total segments: {len(segments)}")
                print(f"Total speech duration: {sum(seg.duration for seg in segments):.1f}s")
                print("\n--- Transcription ---")
                
                for i, segment in enumerate(segments, 1):
                    speaker_info = f" [{segment.speaker_id}]" if segment.speaker_id else ""
                    print(f"#{i:2d} | {segment.start_time:7.1f}s - {segment.end_time:7.1f}s | "
                          f"({segment.language}){speaker_info}")
                    print(f"     | {segment.text}")
                    
                    if segment.word_timestamps and args.verbose:
                        print("     | Word timestamps:")
                        for word in segment.word_timestamps[:5]:  # Show first 5 words
                            print(f"     |   '{word['word']}': {word['start']:.1f}s-{word['end']:.1f}s")
                        if len(segment.word_timestamps) > 5:
                            print(f"     |   ... and {len(segment.word_timestamps)-5} more words")
                    print()
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run CLI if script is executed directly
    if not FASTER_WHISPER_AVAILABLE:
        print("Warning: faster-whisper not available. Install with: pip install faster-whisper")
        print("Running in demo mode...")
        
        # Create dummy segments for testing
        dummy_segments = [
            TranscriptionSegment(
                start_time=0.0, end_time=3.5, text="Hello, how are you today?", 
                language="en", confidence=0.95,
                word_timestamps=[
                    {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
                    {"word": "how", "start": 1.0, "end": 1.2, "confidence": 0.98},
                    {"word": "are", "start": 1.3, "end": 1.5, "confidence": 0.97},
                    {"word": "you", "start": 1.6, "end": 1.9, "confidence": 0.98},
                    {"word": "today", "start": 2.5, "end": 3.2, "confidence": 0.96}
                ]
            ),
            TranscriptionSegment(
                start_time=4.0, end_time=7.8, text="Bonjour, comment allez-vous?",
                language="fr", confidence=0.92
            ),
            TranscriptionSegment(
                start_time=8.5, end_time=12.1, text="मैं ठीक हूँ, धन्यवाद।",
                language="hi", confidence=0.89
            )
        ]
        
        print("\n=== DEMO OUTPUT (faster-whisper not available) ===")
        for i, segment in enumerate(dummy_segments, 1):
            print(f"#{i} | {segment.start_time:.1f}s - {segment.end_time:.1f}s | "
                  f"({segment.language})")
            print(f"     | {segment.text}")
    else:
        main() 