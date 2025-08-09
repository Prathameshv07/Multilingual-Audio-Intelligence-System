"""
Main Pipeline Orchestrator for Multilingual Audio Intelligence System

This module provides the complete end-to-end pipeline orchestration,
integrating audio preprocessing, speaker diarization, speech recognition,
neural machine translation, and output formatting into a unified system.

Key Features:
- Complete end-to-end pipeline execution
- Performance monitoring and benchmarking
- Robust error handling and recovery
- Progress tracking for long operations
- Multiple output format generation
- Command-line interface for batch processing
- Integration with all system modules

Usage:
    python main.py input_audio.wav --output-dir results/
    python main.py audio.mp3 --format json --translate-to en
    python main.py --benchmark test_audio/ --verbose

Dependencies: All src modules, argparse, logging
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all our modules
from audio_processor import AudioProcessor
from speaker_diarizer import SpeakerDiarizer, SpeakerSegment
from speech_recognizer import SpeechRecognizer, TranscriptionSegment
from translator import NeuralTranslator, TranslationResult
from output_formatter import OutputFormatter, ProcessedSegment
from utils import (
    performance_monitor, ProgressTracker, validate_audio_file,
    get_system_info, format_duration, ensure_directory, get_file_info,
    safe_filename
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioIntelligencePipeline:
    """
    Complete multilingual audio intelligence pipeline.
    
    Orchestrates the entire workflow from raw audio input to structured,
    multilingual output with speaker attribution and translations.
    """
    
    def __init__(self,
                 whisper_model_size: str = "small",
                 target_language: str = "en",
                 device: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the complete audio intelligence pipeline.
        
        Args:
            whisper_model_size (str): Whisper model size for ASR
            target_language (str): Target language for translation
            device (str, optional): Device to run on ('cpu', 'cuda', 'auto')
            hf_token (str, optional): Hugging Face token for gated models
            output_dir (str, optional): Directory for output files
        """
        self.whisper_model_size = whisper_model_size
        self.target_language = target_language
        self.device = device
        self.hf_token = hf_token
        self.output_dir = Path(output_dir) if output_dir else Path("./results")
        
        # Ensure output directory exists
        ensure_directory(self.output_dir)
        
        # Initialize pipeline components
        self.audio_processor = None
        self.speaker_diarizer = None
        self.speech_recognizer = None
        self.translator = None
        self.output_formatter = None
        
        # Performance tracking
        self.total_processing_time = 0
        self.component_times = {}
        
        logger.info(f"Initialized AudioIntelligencePipeline:")
        logger.info(f"  - Whisper model: {whisper_model_size}")
        logger.info(f"  - Target language: {target_language}")
        logger.info(f"  - Device: {device or 'auto'}")
        logger.info(f"  - Output directory: {self.output_dir}")
    
    def _initialize_components(self):
        """Lazy initialization of pipeline components."""
        if self.audio_processor is None:
            logger.info("Initializing AudioProcessor...")
            self.audio_processor = AudioProcessor()
        
        if self.speaker_diarizer is None:
            logger.info("Initializing SpeakerDiarizer...")
            self.speaker_diarizer = SpeakerDiarizer(
                hf_token=self.hf_token,
                device=self.device
            )
        
        if self.speech_recognizer is None:
            logger.info("Initializing SpeechRecognizer...")
            self.speech_recognizer = SpeechRecognizer(
                model_size=self.whisper_model_size,
                device=self.device
            )
        
        if self.translator is None:
            logger.info("Initializing NeuralTranslator...")
            self.translator = NeuralTranslator(
                target_language=self.target_language,
                device=self.device
            )
        
        if self.output_formatter is None:
            self.output_formatter = OutputFormatter()
    
    def process_audio(self, 
                     audio_input: str,
                     save_outputs: bool = True,
                     output_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process audio file through complete pipeline.
        
        Args:
            audio_input (str): Path to input audio file
            save_outputs (bool): Whether to save outputs to files
            output_formats (List[str], optional): Formats to generate
            
        Returns:
            Dict[str, Any]: Complete processing results and metadata
        """
        start_time = time.time()
        audio_path = Path(audio_input)
        
        if output_formats is None:
            output_formats = ['json', 'srt', 'text', 'summary']
        
        logger.info(f"Starting audio processing pipeline for: {audio_path.name}")
        
        # Validate input file
        validation = validate_audio_file(audio_path)
        if not validation['valid']:
            raise ValueError(f"Invalid audio file: {validation['error']}")
        
        # Initialize components
        self._initialize_components()
        
        try:
            # Create progress tracker
            progress = ProgressTracker(5, f"Processing {audio_path.name}")
            
            # Step 1: Audio Preprocessing
            progress.update()
            logger.info("Step 1/5: Audio preprocessing...")
            with performance_monitor("audio_preprocessing") as metrics:
                processed_audio, sample_rate = self.audio_processor.process_audio(str(audio_path))
                audio_metadata = self.audio_processor.get_audio_info(str(audio_path))
            
            self.component_times['audio_preprocessing'] = metrics.duration
            logger.info(f"Audio preprocessed: {processed_audio.shape}, {sample_rate}Hz")
            
            # Step 2: Speaker Diarization
            progress.update()
            logger.info("Step 2/5: Speaker diarization...")
            with performance_monitor("speaker_diarization") as metrics:
                speaker_segments = self.speaker_diarizer.diarize(processed_audio, sample_rate)
            
            self.component_times['speaker_diarization'] = metrics.duration
            logger.info(f"Identified {len(set(seg.speaker_id for seg in speaker_segments))} speakers "
                       f"in {len(speaker_segments)} segments")
            
            # Step 3: Speech Recognition
            progress.update()
            logger.info("Step 3/5: Speech recognition...")
            with performance_monitor("speech_recognition") as metrics:
                # Convert speaker segments to format expected by speech recognizer
                speaker_tuples = [(seg.start_time, seg.end_time, seg.speaker_id) 
                                for seg in speaker_segments]
                transcription_segments = self.speech_recognizer.transcribe_segments(
                    processed_audio, sample_rate, speaker_tuples, word_timestamps=True
                )
            
            self.component_times['speech_recognition'] = metrics.duration
            languages_detected = set(seg.language for seg in transcription_segments)
            logger.info(f"Transcribed {len(transcription_segments)} segments, "
                       f"languages: {', '.join(languages_detected)}")
            
            # Step 4: Neural Machine Translation
            progress.update()
            logger.info("Step 4/5: Neural machine translation...")
            with performance_monitor("translation") as metrics:
                translation_results = []
                
                # Group by language for efficient batch translation
                language_groups = {}
                for seg in transcription_segments:
                    if seg.language not in language_groups:
                        language_groups[seg.language] = []
                    language_groups[seg.language].append(seg)
                
                # Translate each language group
                for lang, segments in language_groups.items():
                    if lang != self.target_language:
                        texts = [seg.text for seg in segments]
                        batch_results = self.translator.translate_batch(
                            texts, [lang] * len(texts), self.target_language
                        )
                        translation_results.extend(batch_results)
                    else:
                        # Create identity translations for target language
                        for seg in segments:
                            translation_results.append(TranslationResult(
                                original_text=seg.text,
                                translated_text=seg.text,
                                source_language=lang,
                                target_language=self.target_language,
                                confidence=1.0,
                                model_used="identity"
                            ))
            
            self.component_times['translation'] = metrics.duration
            logger.info(f"Translated {len(translation_results)} text segments")
            
            # Step 5: Output Formatting
            progress.update()
            logger.info("Step 5/5: Output formatting...")
            with performance_monitor("output_formatting") as metrics:
                # Combine all results into ProcessedSegment objects
                processed_segments = self._combine_results(
                    speaker_segments, transcription_segments, translation_results
                )
                
                # Generate outputs
                self.output_formatter = OutputFormatter(audio_path.name)
                all_outputs = self.output_formatter.format_all_outputs(
                    processed_segments, 
                    audio_metadata,
                    self.component_times
                )
            
            self.component_times['output_formatting'] = metrics.duration
            progress.finish()
            
            # Calculate total processing time
            self.total_processing_time = time.time() - start_time
            
            # Save outputs if requested
            if save_outputs:
                saved_files = self._save_outputs(all_outputs, audio_path, output_formats)
            else:
                saved_files = {}
            
            # Prepare final results
            results = {
                'success': True,
                'input_file': str(audio_path),
                'audio_metadata': audio_metadata,
                'processing_stats': {
                    'total_time': self.total_processing_time,
                    'component_times': self.component_times,
                    'num_speakers': len(set(seg.speaker_id for seg in processed_segments)),
                    'num_segments': len(processed_segments),
                    'languages_detected': list(languages_detected),
                    'total_speech_duration': sum(seg.duration for seg in processed_segments)
                },
                'outputs': all_outputs,
                'saved_files': saved_files,
                'processed_segments': processed_segments
            }
            
            logger.info(f"Pipeline completed successfully in {format_duration(self.total_processing_time)}")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _combine_results(self,
                        speaker_segments: List[SpeakerSegment],
                        transcription_segments: List[TranscriptionSegment],
                        translation_results: List[TranslationResult]) -> List[ProcessedSegment]:
        """Combine results from all pipeline stages into unified segments."""
        processed_segments = []
        
        # Create a mapping of speaker segments to transcription/translation
        for i, speaker_seg in enumerate(speaker_segments):
            # Find corresponding transcription segment
            transcription_seg = None
            if i < len(transcription_segments):
                transcription_seg = transcription_segments[i]
            
            # Find corresponding translation result
            translation_result = None
            if i < len(translation_results):
                translation_result = translation_results[i]
            
            # Create ProcessedSegment
            processed_segment = ProcessedSegment(
                start_time=speaker_seg.start_time,
                end_time=speaker_seg.end_time,
                speaker_id=speaker_seg.speaker_id,
                original_text=transcription_seg.text if transcription_seg else "",
                original_language=transcription_seg.language if transcription_seg else "unknown",
                translated_text=translation_result.translated_text if translation_result else "",
                confidence_diarization=speaker_seg.confidence,
                confidence_transcription=transcription_seg.confidence if transcription_seg else 0.0,
                confidence_translation=translation_result.confidence if translation_result else 0.0,
                word_timestamps=transcription_seg.word_timestamps if transcription_seg else None,
                model_info={
                    'diarization_model': 'pyannote/speaker-diarization-3.1',
                    'transcription_model': f'faster-whisper-{self.whisper_model_size}',
                    'translation_model': translation_result.model_used if translation_result else 'none'
                }
            )
            
            processed_segments.append(processed_segment)
        
        return processed_segments
    
    def _save_outputs(self,
                     outputs: Dict[str, str],
                     audio_path: Path,
                     formats: List[str]) -> Dict[str, str]:
        """Save output files to disk."""
        saved_files = {}
        base_filename = safe_filename(audio_path.stem)
        
        format_extensions = {
            'json': 'json',
            'srt_original': 'srt',
            'srt_translated': 'en.srt',
            'text': 'txt',
            'csv': 'csv',
            'timeline': 'timeline.json',
            'summary': 'summary.txt'
        }
        
        for format_name in formats:
            if format_name in outputs:
                extension = format_extensions.get(format_name, 'txt')
                filename = f"{base_filename}.{extension}"
                filepath = self.output_dir / filename
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(outputs[format_name])
                    
                    saved_files[format_name] = str(filepath)
                    logger.info(f"Saved {format_name} output to: {filepath}")
                    
                except Exception as e:
                    logger.error(f"Failed to save {format_name} output: {e}")
        
        return saved_files
    
    def benchmark_system(self, test_audio_path: str) -> Dict[str, Any]:
        """Run system benchmark on test audio."""
        logger.info("Running system benchmark...")
        
        system_info = get_system_info()
        
        # Run multiple iterations for more accurate timing
        iterations = 3
        benchmark_results = []
        
        for i in range(iterations):
            logger.info(f"Benchmark iteration {i+1}/{iterations}")
            try:
                result = self.process_audio(test_audio_path, save_outputs=False)
                benchmark_results.append(result['processing_stats'])
            except Exception as e:
                logger.error(f"Benchmark iteration {i+1} failed: {e}")
                continue
        
        if not benchmark_results:
            return {'error': 'All benchmark iterations failed'}
        
        # Calculate averages
        avg_times = {}
        for component in benchmark_results[0]['component_times']:
            avg_times[component] = sum(r['component_times'][component] for r in benchmark_results) / len(benchmark_results)
        
        avg_total_time = sum(r['total_time'] for r in benchmark_results) / len(benchmark_results)
        
        return {
            'system_info': system_info,
            'test_file': test_audio_path,
            'iterations': len(benchmark_results),
            'average_times': avg_times,
            'average_total_time': avg_total_time,
            'all_iterations': benchmark_results
        }


def main():
    """Command-line interface for the audio intelligence pipeline."""
    parser = argparse.ArgumentParser(
        description="Multilingual Audio Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py audio.wav                     # Process with defaults
  python main.py audio.mp3 --output-dir ./out  # Custom output directory
  python main.py audio.flac --translate-to es  # Translate to Spanish
  python main.py --benchmark test.wav          # Run performance benchmark
  python main.py audio.ogg --format json text  # Generate specific formats
        """
    )
    
    # Input arguments
    parser.add_argument("audio_file", nargs='?', help="Path to input audio file")
    
    # Model configuration
    parser.add_argument("--whisper-model", choices=["tiny", "small", "medium", "large"],
                       default="small", help="Whisper model size (default: small)")
    parser.add_argument("--translate-to", default="en",
                       help="Target language for translation (default: en)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to run on (default: auto)")
    parser.add_argument("--hf-token", help="Hugging Face token for gated models")
    
    # Output configuration
    parser.add_argument("--output-dir", "-o", default="./results",
                       help="Output directory (default: ./results)")
    parser.add_argument("--format", nargs='+',
                       choices=["json", "srt", "text", "csv", "timeline", "summary", "all"],
                       default=["json", "srt", "text", "summary"],
                       help="Output formats to generate")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save outputs to files")
    
    # Utility options
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--system-info", action="store_true",
                       help="Show system information and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-error output")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Handle system info request
    if args.system_info:
        system_info = get_system_info()
        print("\n=== SYSTEM INFORMATION ===")
        for key, value in system_info.items():
            print(f"{key}: {value}")
        return
    
    # Validate audio file argument
    if not args.audio_file:
        parser.error("Audio file is required (unless using --system-info)")
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")
    
    try:
        # Initialize pipeline
        pipeline = AudioIntelligencePipeline(
            whisper_model_size=args.whisper_model,
            target_language=args.translate_to,
            device=args.device,
            hf_token=args.hf_token,
            output_dir=args.output_dir
        )
        
        if args.benchmark:
            # Run benchmark
            print(f"\n=== RUNNING BENCHMARK ON {audio_path.name} ===")
            benchmark_results = pipeline.benchmark_system(str(audio_path))
            
            if 'error' in benchmark_results:
                print(f"Benchmark failed: {benchmark_results['error']}")
                return 1
            
            print(f"\nBenchmark Results ({benchmark_results['iterations']} iterations):")
            print(f"Average total time: {format_duration(benchmark_results['average_total_time'])}")
            print("\nComponent breakdown:")
            for component, avg_time in benchmark_results['average_times'].items():
                print(f"  {component}: {format_duration(avg_time)}")
            
            print(f"\nSystem: {benchmark_results['system_info']['platform']}")
            print(f"GPU: {benchmark_results['system_info']['gpu_info']}")
            
        else:
            # Process audio file
            output_formats = args.format
            if 'all' in output_formats:
                output_formats = ['json', 'srt_original', 'srt_translated', 'text', 'csv', 'timeline', 'summary']
            
            results = pipeline.process_audio(
                str(audio_path),
                save_outputs=not args.no_save,
                output_formats=output_formats
            )
            
            # Print summary
            stats = results['processing_stats']
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"File: {audio_path.name}")
            print(f"Total time: {format_duration(stats['total_time'])}")
            print(f"Speakers: {stats['num_speakers']}")
            print(f"Segments: {stats['num_segments']}")
            print(f"Languages: {', '.join(stats['languages_detected'])}")
            print(f"Speech duration: {format_duration(stats['total_speech_duration'])}")
            
            if results['saved_files']:
                print(f"\nOutput files saved to: {args.output_dir}")
                for format_name, filepath in results['saved_files'].items():
                    print(f"  {format_name}: {Path(filepath).name}")
            
            if not args.quiet:
                # Show sample of results
                segments = results['processed_segments'][:3]  # First 3 segments
                print(f"\nSample output (first {len(segments)} segments):")
                for i, seg in enumerate(segments, 1):
                    speaker = seg.speaker_id.replace("SPEAKER_", "Speaker ")
                    time_str = f"{seg.start_time:.1f}s-{seg.end_time:.1f}s"
                    print(f"  #{i} [{time_str}] {speaker} ({seg.original_language}):")
                    print(f"      Original: {seg.original_text}")
                    if seg.original_language != args.translate_to:
                        print(f"      Translated: {seg.translated_text}")
                
                if len(results['processed_segments']) > 3:
                    print(f"  ... and {len(results['processed_segments']) - 3} more segments")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 