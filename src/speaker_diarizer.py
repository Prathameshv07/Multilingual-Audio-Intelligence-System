"""
Speaker Diarization Module for Multilingual Audio Intelligence System

This module implements state-of-the-art speaker diarization using pyannote.audio.
It segments audio to identify "who spoke when" with high accuracy and language-agnostic
speaker separation capabilities as required by PS-6.

Key Features:
- SOTA speaker diarization using pyannote.audio
- Language-agnostic voice characteristic analysis
- Integrated Voice Activity Detection (VAD)
- Automatic speaker count detection
- CPU and GPU optimization support
- Robust error handling and logging

Model: pyannote/speaker-diarization-3.1
Dependencies: pyannote.audio, torch, transformers
"""

import os
import logging
import warnings
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import tempfile
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logging.warning("pyannote.audio not available. Install with: pip install pyannote.audio")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress various warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class SpeakerSegment:
    """
    Data class representing a single speaker segment.
    
    Attributes:
        start_time (float): Segment start time in seconds
        end_time (float): Segment end time in seconds  
        speaker_id (str): Unique speaker identifier (e.g., "SPEAKER_00")
        confidence (float): Confidence score of the diarization (if available)
    """
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'speaker_id': self.speaker_id,
            'duration': self.duration,
            'confidence': self.confidence
        }


class SpeakerDiarizer:
    """
    State-of-the-art speaker diarization using pyannote.audio.
    
    This class provides language-agnostic speaker diarization capabilities,
    focusing on acoustic voice characteristics rather than linguistic content.
    """
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 hf_token: Optional[str] = None,
                 device: Optional[str] = None,
                 min_speakers: Optional[int] = None,
                 max_speakers: Optional[int] = None):
        """
        Initialize the Speaker Diarizer.
        
        Args:
            model_name (str): Hugging Face model name for diarization
            hf_token (str, optional): Hugging Face token for gated models
            device (str, optional): Device to run on ('cpu', 'cuda', 'auto')
            min_speakers (int, optional): Minimum number of speakers to detect
            max_speakers (int, optional): Maximum number of speakers to detect
        """
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_TOKEN')
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Device selection
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing SpeakerDiarizer on {self.device}")
        
        # Initialize pipeline
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the pyannote.audio diarization pipeline."""
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: pip install pyannote.audio"
            )
        
        try:
            # Load the pre-trained pipeline
            logger.info(f"Loading {self.model_name}...")
            
            if self.hf_token:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name, 
                    use_auth_token=self.hf_token
                )
            else:
                # Try without token first (for public models)
                try:
                    self.pipeline = Pipeline.from_pretrained(self.model_name)
                except Exception as e:
                    logger.error(
                        f"Failed to load {self.model_name}. "
                        "This model may be gated and require a Hugging Face token. "
                        f"Set HUGGINGFACE_TOKEN environment variable. Error: {e}"
                    )
                    raise
            
            # Move pipeline to appropriate device
            self.pipeline = self.pipeline.to(self.device)
            
            # Configure speaker count constraints
            if self.min_speakers is not None or self.max_speakers is not None:
                self.pipeline.instantiate({
                    "clustering": {
                        "min_cluster_size": self.min_speakers or 1,
                        "max_num_speakers": self.max_speakers or 20
                    }
                })
            
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise
    
    def diarize(self, 
                audio_input: Union[str, np.ndarray], 
                sample_rate: int = 16000) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio input.
        
        Args:
            audio_input: Audio file path or numpy array
            sample_rate: Sample rate if audio_input is numpy array
            
        Returns:
            List[SpeakerSegment]: List of speaker segments with timestamps
            
        Raises:
            ValueError: If input is invalid
            Exception: For diarization errors
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call _load_pipeline() first.")
        
        try:
            # Prepare audio input for pyannote
            audio_file = self._prepare_audio_input(audio_input, sample_rate)
            
            logger.info("Starting speaker diarization...")
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Run diarization
            diarization_result = self.pipeline(audio_file)
            
            if end_time and start_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
                logger.info(f"Diarization completed in {processing_time:.2f}s")
            
            # Convert results to structured format
            segments = self._parse_diarization_result(diarization_result)
            
            # Log summary
            num_speakers = len(set(seg.speaker_id for seg in segments))
            total_speech_time = sum(seg.duration for seg in segments)
            
            logger.info(f"Detected {num_speakers} speakers, {len(segments)} segments, "
                       f"{total_speech_time:.1f}s total speech")
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            raise
        
        finally:
            # Clean up temporary files if created
            if isinstance(audio_input, np.ndarray):
                try:
                    if hasattr(audio_file, 'name') and os.path.exists(audio_file.name):
                        os.unlink(audio_file.name)
                except Exception:
                    pass
    
    def _prepare_audio_input(self, 
                           audio_input: Union[str, np.ndarray], 
                           sample_rate: int) -> str:
        """
        Prepare audio input for pyannote.audio pipeline.
        
        Args:
            audio_input: Audio file path or numpy array
            sample_rate: Sample rate for numpy array input
            
        Returns:
            str: Path to audio file ready for pyannote
        """
        if isinstance(audio_input, str):
            # File path - validate existence
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return audio_input
            
        elif isinstance(audio_input, np.ndarray):
            # Numpy array - save to temporary file
            return self._save_array_to_tempfile(audio_input, sample_rate)
            
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
    
    def _save_array_to_tempfile(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """
        Save numpy array to temporary WAV file for pyannote processing.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            str: Path to temporary WAV file
        """
        try:
            import soundfile as sf
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix='.wav',
                prefix='diarization_'
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Ensure audio is in correct format
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
            
            # Normalize to prevent clipping
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Save using soundfile
            sf.write(temp_path, audio_array, sample_rate)
            
            logger.debug(f"Saved audio array to temporary file: {temp_path}")
            return temp_path
            
        except ImportError:
            # Fallback to scipy if soundfile not available
            try:
                from scipy.io import wavfile
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.wav',
                    prefix='diarization_'
                )
                temp_path = temp_file.name
                temp_file.close()
                
                # Convert to 16-bit int for scipy
                if audio_array.dtype != np.int16:
                    audio_array_int = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array_int = audio_array
                
                wavfile.write(temp_path, sample_rate, audio_array_int)
                
                logger.debug(f"Saved audio array using scipy: {temp_path}")
                return temp_path
                
            except ImportError:
                raise ImportError(
                    "Neither soundfile nor scipy available for audio saving. "
                    "Install with: pip install soundfile"
                )
    
    def _parse_diarization_result(self, diarization: Annotation) -> List[SpeakerSegment]:
        """
        Parse pyannote diarization result into structured segments.
        
        Args:
            diarization: pyannote Annotation object
            
        Returns:
            List[SpeakerSegment]: Parsed speaker segments
        """
        segments = []
        
        for segment, _, speaker_label in diarization.itertracks(yield_label=True):
            # Convert pyannote segment to our format
            speaker_segment = SpeakerSegment(
                start_time=float(segment.start),
                end_time=float(segment.end),
                speaker_id=str(speaker_label),
                confidence=1.0  # pyannote doesn't provide segment-level confidence
            )
            segments.append(speaker_segment)
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        
        return segments
    
    def get_speaker_statistics(self, segments: List[SpeakerSegment]) -> Dict[str, dict]:
        """
        Generate speaker statistics from diarization results.
        
        Args:
            segments: List of speaker segments
            
        Returns:
            Dict: Speaker statistics including speaking time, turn counts, etc.
        """
        stats = {}
        
        for segment in segments:
            speaker_id = segment.speaker_id
            
            if speaker_id not in stats:
                stats[speaker_id] = {
                    'total_speaking_time': 0.0,
                    'number_of_turns': 0,
                    'average_turn_duration': 0.0,
                    'longest_turn': 0.0,
                    'shortest_turn': float('inf')
                }
            
            # Update statistics
            stats[speaker_id]['total_speaking_time'] += segment.duration
            stats[speaker_id]['number_of_turns'] += 1
            stats[speaker_id]['longest_turn'] = max(
                stats[speaker_id]['longest_turn'], 
                segment.duration
            )
            stats[speaker_id]['shortest_turn'] = min(
                stats[speaker_id]['shortest_turn'], 
                segment.duration
            )
        
        # Calculate averages
        for speaker_id, speaker_stats in stats.items():
            if speaker_stats['number_of_turns'] > 0:
                speaker_stats['average_turn_duration'] = (
                    speaker_stats['total_speaking_time'] / 
                    speaker_stats['number_of_turns']
                )
            
            # Handle edge case for shortest turn
            if speaker_stats['shortest_turn'] == float('inf'):
                speaker_stats['shortest_turn'] = 0.0
        
        return stats
    
    def merge_short_segments(self, 
                           segments: List[SpeakerSegment], 
                           min_duration: float = 1.0) -> List[SpeakerSegment]:
        """
        Merge segments that are too short with adjacent segments from same speaker.
        
        Args:
            segments: List of speaker segments
            min_duration: Minimum duration for segments in seconds
            
        Returns:
            List[SpeakerSegment]: Processed segments with short ones merged
        """
        if not segments:
            return segments
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # If current segment is too short and next is same speaker, merge
            if (current_segment.duration < min_duration and 
                current_segment.speaker_id == next_segment.speaker_id):
                
                # Extend current segment to include next segment
                current_segment.end_time = next_segment.end_time
                
            else:
                # Add current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged_segments.append(current_segment)
        
        logger.debug(f"Merged {len(segments)} segments into {len(merged_segments)}")
        
        return merged_segments
    
    def export_to_rttm(self, 
                       segments: List[SpeakerSegment], 
                       audio_filename: str = "audio") -> str:
        """
        Export diarization results to RTTM format.
        
        RTTM (Rich Transcription Time Marked) is a standard format
        for speaker diarization results.
        
        Args:
            segments: List of speaker segments
            audio_filename: Name of the audio file for RTTM output
            
        Returns:
            str: RTTM formatted string
        """
        rttm_lines = []
        
        for segment in segments:
            # RTTM format: SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf>
            rttm_line = (
                f"SPEAKER {audio_filename} 1 "
                f"{segment.start_time:.3f} {segment.duration:.3f} "
                f"<NA> <NA> {segment.speaker_id} {segment.confidence:.3f}"
            )
            rttm_lines.append(rttm_line)
        
        return "\n".join(rttm_lines)
    
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        # Clear GPU cache if using CUDA
        if hasattr(self, 'device') and self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


# Convenience function for easy usage
def diarize_audio(audio_input: Union[str, np.ndarray], 
                  sample_rate: int = 16000,
                  hf_token: Optional[str] = None,
                  min_speakers: Optional[int] = None,
                  max_speakers: Optional[int] = None,
                  merge_short: bool = True,
                  min_duration: float = 1.0) -> List[SpeakerSegment]:
    """
    Convenience function to perform speaker diarization with default settings.
    
    Args:
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate for numpy array input
        hf_token: Hugging Face token for gated models
        min_speakers: Minimum number of speakers to detect
        max_speakers: Maximum number of speakers to detect
        merge_short: Whether to merge short segments
        min_duration: Minimum duration for segments (if merge_short=True)
        
    Returns:
        List[SpeakerSegment]: Speaker diarization results
        
    Example:
        >>> # From file
        >>> segments = diarize_audio("meeting.wav")
        >>> 
        >>> # From numpy array
        >>> import numpy as np
        >>> audio_data = np.random.randn(16000 * 60)  # 1 minute of audio
        >>> segments = diarize_audio(audio_data, sample_rate=16000)
        >>> 
        >>> # Print results
        >>> for seg in segments:
        >>>     print(f"{seg.speaker_id}: {seg.start_time:.1f}s - {seg.end_time:.1f}s")
    """
    # Initialize diarizer
    diarizer = SpeakerDiarizer(
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # Perform diarization
    segments = diarizer.diarize(audio_input, sample_rate)
    
    # Merge short segments if requested
    if merge_short and segments:
        segments = diarizer.merge_short_segments(segments, min_duration)
    
    return segments


# Example usage and testing
if __name__ == "__main__":
    import sys
    import argparse
    import json
    
    def main():
        """Command line interface for testing speaker diarization."""
        parser = argparse.ArgumentParser(description="Speaker Diarization Tool")
        parser.add_argument("audio_file", help="Path to audio file")
        parser.add_argument("--token", help="Hugging Face token")
        parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
        parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers") 
        parser.add_argument("--output-format", choices=["json", "rttm", "text"], 
                          default="text", help="Output format")
        parser.add_argument("--merge-short", action="store_true", 
                          help="Merge short segments")
        parser.add_argument("--min-duration", type=float, default=1.0,
                          help="Minimum segment duration for merging")
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose logging")
        
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            # Perform diarization
            print(f"Processing audio file: {args.audio_file}")
            
            segments = diarize_audio(
                audio_input=args.audio_file,
                hf_token=args.token,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                merge_short=args.merge_short,
                min_duration=args.min_duration
            )
            
            # Output results in requested format
            if args.output_format == "json":
                # JSON output
                result = {
                    "audio_file": args.audio_file,
                    "num_speakers": len(set(seg.speaker_id for seg in segments)),
                    "num_segments": len(segments),
                    "total_speech_time": sum(seg.duration for seg in segments),
                    "segments": [seg.to_dict() for seg in segments]
                }
                print(json.dumps(result, indent=2))
                
            elif args.output_format == "rttm":
                # RTTM output
                diarizer = SpeakerDiarizer()
                rttm_content = diarizer.export_to_rttm(segments, args.audio_file)
                print(rttm_content)
                
            else:  # text format
                # Human-readable text output
                print(f"\n=== SPEAKER DIARIZATION RESULTS ===")
                print(f"Audio file: {args.audio_file}")
                print(f"Number of speakers: {len(set(seg.speaker_id for seg in segments))}")
                print(f"Number of segments: {len(segments)}")
                print(f"Total speech time: {sum(seg.duration for seg in segments):.1f}s")
                print("\n--- Segment Details ---")
                
                for i, segment in enumerate(segments, 1):
                    print(f"#{i:2d} | {segment.speaker_id:10s} | "
                          f"{segment.start_time:7.1f}s - {segment.end_time:7.1f}s | "
                          f"{segment.duration:5.1f}s")
                
                # Speaker statistics
                diarizer = SpeakerDiarizer()
                stats = diarizer.get_speaker_statistics(segments)
                
                print("\n--- Speaker Statistics ---")
                for speaker_id, speaker_stats in stats.items():
                    print(f"{speaker_id}:")
                    print(f"  Speaking time: {speaker_stats['total_speaking_time']:.1f}s")
                    print(f"  Number of turns: {speaker_stats['number_of_turns']}")
                    print(f"  Average turn: {speaker_stats['average_turn_duration']:.1f}s")
                    print(f"  Longest turn: {speaker_stats['longest_turn']:.1f}s")
                    print(f"  Shortest turn: {speaker_stats['shortest_turn']:.1f}s")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run CLI if script is executed directly
    if not PYANNOTE_AVAILABLE:
        print("Warning: pyannote.audio not available. Install with: pip install pyannote.audio")
        print("Running in demo mode...")
        
        # Create dummy segments for testing
        dummy_segments = [
            SpeakerSegment(0.0, 5.2, "SPEAKER_00", 0.95),
            SpeakerSegment(5.5, 8.3, "SPEAKER_01", 0.87),
            SpeakerSegment(8.8, 12.1, "SPEAKER_00", 0.92),
            SpeakerSegment(12.5, 15.7, "SPEAKER_01", 0.89),
        ]
        
        print("\n=== DEMO OUTPUT (pyannote.audio not available) ===")
        for segment in dummy_segments:
            print(f"{segment.speaker_id}: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
    else:
        main()