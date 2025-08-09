"""
Output Formatting Module for Multilingual Audio Intelligence System

This module consolidates processed data from speaker diarization, speech recognition,
and neural machine translation into various structured formats for different use cases.
Designed for maximum flexibility and user-friendly output presentation.

Key Features:
- JSON format for programmatic access and API integration
- SRT subtitle format for video/media players with speaker labels
- Human-readable text format with rich metadata
- Interactive timeline format for web visualization
- CSV export for data analysis and spreadsheet applications
- Rich metadata preservation throughout all formats
- Error handling and graceful degradation

Output Formats: JSON, SRT, Plain Text, CSV, Timeline
Dependencies: json, csv, dataclasses
"""

import json
import csv
import io
import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import timedelta
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedSegment:
    """
    Unified data structure for a processed audio segment with all metadata.
    
    Attributes:
        start_time (float): Segment start time in seconds
        end_time (float): Segment end time in seconds
        speaker_id (str): Speaker identifier
        original_text (str): Transcribed text in original language
        original_language (str): Detected original language code
        translated_text (str): English translation
        confidence_diarization (float): Speaker diarization confidence
        confidence_transcription (float): Speech recognition confidence
        confidence_translation (float): Translation confidence
        word_timestamps (List[Dict]): Word-level timing information
        model_info (Dict): Information about models used
    """
    start_time: float
    end_time: float
    speaker_id: str
    original_text: str
    original_language: str
    translated_text: str
    confidence_diarization: float = 1.0
    confidence_transcription: float = 1.0
    confidence_translation: float = 1.0
    word_timestamps: Optional[List[Dict]] = None
    model_info: Optional[Dict] = None
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class OutputFormatter:
    """
    Advanced output formatting for multilingual audio intelligence results.
    
    Converts processed audio data into multiple user-friendly formats with
    comprehensive metadata and beautiful presentation.
    """
    
    def __init__(self, audio_filename: str = "audio_file"):
        """
        Initialize the Output Formatter.
        
        Args:
            audio_filename (str): Name of the original audio file for references
        """
        self.audio_filename = audio_filename
        self.creation_timestamp = None
        self.processing_stats = {}
    
    def format_all_outputs(self,
                          segments: List[ProcessedSegment],
                          audio_metadata: Optional[Dict] = None,
                          processing_stats: Optional[Dict] = None) -> Dict[str, str]:
        """
        Generate all output formats in one call.
        
        Args:
            segments (List[ProcessedSegment]): Processed audio segments
            audio_metadata (Dict, optional): Original audio file metadata
            processing_stats (Dict, optional): Processing time and performance stats
            
        Returns:
            Dict[str, str]: Dictionary with all formatted outputs
        """
        self.processing_stats = processing_stats or {}
        
        return {
            'json': self.to_json(segments, audio_metadata),
            'srt_original': self.to_srt(segments, use_translation=False),
            'srt_translated': self.to_srt(segments, use_translation=True),
            'text': self.to_text(segments, audio_metadata),
            'csv': self.to_csv(segments),
            'timeline': self.to_timeline_json(segments),
            'summary': self.generate_summary(segments, audio_metadata)
        }
    
    def to_json(self,
               segments: List[ProcessedSegment],
               audio_metadata: Optional[Dict] = None) -> str:
        """
        Convert segments to comprehensive JSON format.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            audio_metadata (Dict, optional): Audio file metadata
            
        Returns:
            str: JSON formatted string
        """
        # Generate comprehensive statistics
        stats = self._generate_statistics(segments)
        
        # Create the main JSON structure
        output = {
            "metadata": {
                "audio_filename": self.audio_filename,
                "processing_timestamp": self._get_timestamp(),
                "total_segments": len(segments),
                "total_speakers": len(set(seg.speaker_id for seg in segments)),
                "languages_detected": list(set(seg.original_language for seg in segments)),
                "total_audio_duration": stats['total_duration'],
                "total_speech_duration": stats['total_speech_duration'],
                "speech_ratio": stats['speech_ratio'],
                "audio_metadata": audio_metadata,
                "processing_stats": self.processing_stats
            },
            "statistics": stats,
            "segments": [seg.to_dict() for seg in segments],
            "speakers": self._generate_speaker_stats(segments),
            "languages": self._generate_language_stats(segments)
        }
        
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    def to_srt(self,
              segments: List[ProcessedSegment],
              use_translation: bool = False,
              include_speaker_labels: bool = True) -> str:
        """
        Convert segments to SRT subtitle format.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            use_translation (bool): Use translated text instead of original
            include_speaker_labels (bool): Include speaker names in subtitles
            
        Returns:
            str: SRT formatted string
        """
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            # Format timestamp for SRT (HH:MM:SS,mmm)
            start_time = self._seconds_to_srt_time(segment.start_time)
            end_time = self._seconds_to_srt_time(segment.end_time)
            
            # Choose text based on preference
            text = segment.translated_text if use_translation else segment.original_text
            
            # Add speaker label if requested
            if include_speaker_labels:
                speaker_name = self._format_speaker_name(segment.speaker_id)
                text = f"<v {speaker_name}>{text}"
            
            # Add language indicator for original text
            if not use_translation and segment.original_language != 'en':
                text = f"[{segment.original_language.upper()}] {text}"
            
            # Build SRT entry
            srt_entry = [
                str(i),
                f"{start_time} --> {end_time}",
                text,
                ""  # Empty line separator
            ]
            
            srt_lines.extend(srt_entry)
        
        return "\n".join(srt_lines)
    
    def to_text(self,
               segments: List[ProcessedSegment],
               audio_metadata: Optional[Dict] = None,
               include_word_timestamps: bool = False) -> str:
        """
        Convert segments to human-readable text format.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            audio_metadata (Dict, optional): Audio file metadata
            include_word_timestamps (bool): Include detailed word timing
            
        Returns:
            str: Formatted text string
        """
        lines = []
        
        # Header section
        lines.append("=" * 80)
        lines.append("MULTILINGUAL AUDIO INTELLIGENCE ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        # File information
        lines.append(f"Audio File: {self.audio_filename}")
        lines.append(f"Analysis Date: {self._get_timestamp()}")
        
        if audio_metadata:
            lines.append(f"Duration: {self._format_duration(audio_metadata.get('duration_seconds', 0))}")
            lines.append(f"Sample Rate: {audio_metadata.get('sample_rate', 'Unknown')} Hz")
            lines.append(f"Channels: {audio_metadata.get('channels', 'Unknown')}")
        
        lines.append("")
        
        # Statistics section
        stats = self._generate_statistics(segments)
        lines.append("ANALYSIS SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Speakers: {len(set(seg.speaker_id for seg in segments))}")
        lines.append(f"Languages Detected: {', '.join(set(seg.original_language for seg in segments))}")
        lines.append(f"Total Segments: {len(segments)}")
        lines.append(f"Speech Duration: {self._format_duration(stats['total_speech_duration'])}")
        lines.append(f"Speech Ratio: {stats['speech_ratio']:.1%}")
        
        if self.processing_stats:
            lines.append(f"Processing Time: {self.processing_stats.get('total_time', 'Unknown')}")
        
        lines.append("")
        
        # Speaker statistics
        speaker_stats = self._generate_speaker_stats(segments)
        lines.append("SPEAKER BREAKDOWN")
        lines.append("-" * 40)
        
        for speaker_id, stats in speaker_stats.items():
            speaker_name = self._format_speaker_name(speaker_id)
            lines.append(f"{speaker_name}:")
            lines.append(f"  Speaking Time: {self._format_duration(stats['total_speaking_time'])}")
            lines.append(f"  Number of Turns: {stats['number_of_turns']}")
            lines.append(f"  Average Turn: {self._format_duration(stats['average_turn_duration'])}")
            lines.append(f"  Longest Turn: {self._format_duration(stats['longest_turn'])}")
            if stats['languages']:
                lines.append(f"  Languages: {', '.join(stats['languages'])}")
        
        lines.append("")
        
        # Transcript section
        lines.append("FULL TRANSCRIPT")
        lines.append("=" * 80)
        lines.append("")
        
        for i, segment in enumerate(segments, 1):
            # Timestamp and speaker header
            timestamp = f"[{self._format_duration(segment.start_time)} - {self._format_duration(segment.end_time)}]"
            speaker_name = self._format_speaker_name(segment.speaker_id)
            
            lines.append(f"#{i:3d} {timestamp} {speaker_name}")
            
            # Original text with language indicator
            if segment.original_language != 'en':
                lines.append(f"     Original ({segment.original_language}): {segment.original_text}")
                lines.append(f"     Translation: {segment.translated_text}")
            else:
                lines.append(f"     Text: {segment.original_text}")
            
            # Confidence scores
            lines.append(f"     Confidence: D:{segment.confidence_diarization:.2f} "
                        f"T:{segment.confidence_transcription:.2f} "
                        f"TR:{segment.confidence_translation:.2f}")
            
            # Word timestamps if requested
            if include_word_timestamps and segment.word_timestamps:
                lines.append("     Word Timing:")
                word_lines = []
                for word_info in segment.word_timestamps[:10]:  # Limit to first 10 words
                    word_time = f"{word_info['start']:.1f}s"
                    word_lines.append(f"'{word_info['word']}'@{word_time}")
                
                lines.append(f"       {', '.join(word_lines)}")
                if len(segment.word_timestamps) > 10:
                    lines.append(f"       ... and {len(segment.word_timestamps) - 10} more words")
            
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("Generated by Multilingual Audio Intelligence System")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_csv(self, segments: List[ProcessedSegment]) -> str:
        """
        Convert segments to CSV format for data analysis.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            
        Returns:
            str: CSV formatted string
        """
        output = io.StringIO()
        
        fieldnames = [
            'segment_id', 'start_time', 'end_time', 'duration',
            'speaker_id', 'original_language', 'original_text',
            'translated_text', 'confidence_diarization',
            'confidence_transcription', 'confidence_translation',
            'word_count_original', 'word_count_translated'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, segment in enumerate(segments, 1):
            row = {
                'segment_id': i,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'speaker_id': segment.speaker_id,
                'original_language': segment.original_language,
                'original_text': segment.original_text,
                'translated_text': segment.translated_text,
                'confidence_diarization': segment.confidence_diarization,
                'confidence_transcription': segment.confidence_transcription,
                'confidence_translation': segment.confidence_translation,
                'word_count_original': len(segment.original_text.split()),
                'word_count_translated': len(segment.translated_text.split())
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def to_timeline_json(self, segments: List[ProcessedSegment]) -> str:
        """
        Convert segments to timeline JSON format for interactive visualization.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            
        Returns:
            str: Timeline JSON formatted string
        """
        # Prepare timeline data
        timeline_data = {
            "title": {
                "text": {
                    "headline": f"Audio Analysis: {self.audio_filename}",
                    "text": f"Interactive timeline of speaker segments and transcription"
                }
            },
            "events": []
        }
        
        for i, segment in enumerate(segments):
            event = {
                "start_date": {
                    "second": int(segment.start_time)
                },
                "end_date": {
                    "second": int(segment.end_time)
                },
                "text": {
                    "headline": f"{self._format_speaker_name(segment.speaker_id)} ({segment.original_language})",
                    "text": f"<p><strong>Original:</strong> {segment.original_text}</p>"
                           f"<p><strong>Translation:</strong> {segment.translated_text}</p>"
                           f"<p><em>Duration: {segment.duration:.1f}s, "
                           f"Confidence: {segment.confidence_transcription:.2f}</em></p>"
                },
                "group": segment.speaker_id,
                "media": {
                    "caption": f"Segment {i+1}: {self._format_duration(segment.start_time)} - {self._format_duration(segment.end_time)}"
                }
            }
            
            timeline_data["events"].append(event)
        
        return json.dumps(timeline_data, indent=2, ensure_ascii=False)
    
    def generate_summary(self,
                        segments: List[ProcessedSegment],
                        audio_metadata: Optional[Dict] = None) -> str:
        """
        Generate a concise summary of the analysis.
        
        Args:
            segments (List[ProcessedSegment]): Processed segments
            audio_metadata (Dict, optional): Audio file metadata
            
        Returns:
            str: Summary text
        """
        if not segments:
            return "No speech segments were detected in the audio file."
        
        stats = self._generate_statistics(segments)
        speaker_stats = self._generate_speaker_stats(segments)
        
        summary_lines = []
        
        # Basic overview
        summary_lines.append(f"ANALYSIS SUMMARY FOR {self.audio_filename}")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # Key statistics
        summary_lines.append(f"• {len(set(seg.speaker_id for seg in segments))} speakers detected")
        summary_lines.append(f"• {len(segments)} speech segments identified")
        summary_lines.append(f"• {len(set(seg.original_language for seg in segments))} languages detected: "
                            f"{', '.join(set(seg.original_language for seg in segments))}")
        summary_lines.append(f"• {stats['speech_ratio']:.1%} of audio contains speech")
        summary_lines.append("")
        
        # Speaker overview
        summary_lines.append("SPEAKER BREAKDOWN:")
        for speaker_id, stats in speaker_stats.items():
            speaker_name = self._format_speaker_name(speaker_id)
            percentage = (stats['total_speaking_time'] / sum(s['total_speaking_time'] for s in speaker_stats.values())) * 100
            summary_lines.append(f"• {speaker_name}: {self._format_duration(stats['total_speaking_time'])} "
                                f"({percentage:.1f}%) across {stats['number_of_turns']} turns")
        
        summary_lines.append("")
        
        # Language breakdown if multilingual
        languages = set(seg.original_language for seg in segments)
        if len(languages) > 1:
            summary_lines.append("LANGUAGE BREAKDOWN:")
            lang_stats = self._generate_language_stats(segments)
            for lang, stats in lang_stats.items():
                percentage = (stats['speaking_time'] / sum(s['speaking_time'] for s in lang_stats.values())) * 100
                summary_lines.append(f"• {lang.upper()}: {self._format_duration(stats['speaking_time'])} "
                                    f"({percentage:.1f}%) in {stats['segment_count']} segments")
            summary_lines.append("")
        
        # Key insights
        summary_lines.append("KEY INSIGHTS:")
        
        # Most active speaker
        most_active = max(speaker_stats.items(), key=lambda x: x[1]['total_speaking_time'])
        summary_lines.append(f"• Most active speaker: {self._format_speaker_name(most_active[0])}")
        
        # Longest turn
        longest_segment = max(segments, key=lambda s: s.duration)
        summary_lines.append(f"• Longest speaking turn: {self._format_duration(longest_segment.duration)} "
                            f"by {self._format_speaker_name(longest_segment.speaker_id)}")
        
        # Average confidence
        avg_confidence = sum(seg.confidence_transcription for seg in segments) / len(segments)
        summary_lines.append(f"• Average transcription confidence: {avg_confidence:.2f}")
        
        if len(languages) > 1:
            # Code-switching detection
            code_switches = 0
            for i in range(1, len(segments)):
                if segments[i-1].speaker_id == segments[i].speaker_id and segments[i-1].original_language != segments[i].original_language:
                    code_switches += 1
            if code_switches > 0:
                summary_lines.append(f"• {code_switches} potential code-switching instances detected")
        
        return "\n".join(summary_lines)
    
    def _generate_statistics(self, segments: List[ProcessedSegment]) -> Dict[str, Any]:
        """Generate comprehensive statistics from segments."""
        if not segments:
            return {}
        
        total_speech_duration = sum(seg.duration for seg in segments)
        total_duration = max(seg.end_time for seg in segments) if segments else 0
        
        return {
            'total_duration': total_duration,
            'total_speech_duration': total_speech_duration,
            'speech_ratio': total_speech_duration / total_duration if total_duration > 0 else 0,
            'average_segment_duration': total_speech_duration / len(segments),
            'longest_segment': max(seg.duration for seg in segments),
            'shortest_segment': min(seg.duration for seg in segments),
            'average_confidence_diarization': sum(seg.confidence_diarization for seg in segments) / len(segments),
            'average_confidence_transcription': sum(seg.confidence_transcription for seg in segments) / len(segments),
            'average_confidence_translation': sum(seg.confidence_translation for seg in segments) / len(segments),
            'total_words_original': sum(len(seg.original_text.split()) for seg in segments),
            'total_words_translated': sum(len(seg.translated_text.split()) for seg in segments)
        }
    
    def _generate_speaker_stats(self, segments: List[ProcessedSegment]) -> Dict[str, Dict]:
        """Generate per-speaker statistics."""
        speaker_stats = {}
        
        for segment in segments:
            speaker_id = segment.speaker_id
            
            if speaker_id not in speaker_stats:
                speaker_stats[speaker_id] = {
                    'total_speaking_time': 0.0,
                    'number_of_turns': 0,
                    'longest_turn': 0.0,
                    'shortest_turn': float('inf'),
                    'languages': set()
                }
            
            stats = speaker_stats[speaker_id]
            stats['total_speaking_time'] += segment.duration
            stats['number_of_turns'] += 1
            stats['longest_turn'] = max(stats['longest_turn'], segment.duration)
            stats['shortest_turn'] = min(stats['shortest_turn'], segment.duration)
            stats['languages'].add(segment.original_language)
        
        # Calculate averages and convert sets to lists
        for speaker_id, stats in speaker_stats.items():
            if stats['number_of_turns'] > 0:
                stats['average_turn_duration'] = stats['total_speaking_time'] / stats['number_of_turns']
            else:
                stats['average_turn_duration'] = 0.0
            
            if stats['shortest_turn'] == float('inf'):
                stats['shortest_turn'] = 0.0
            
            stats['languages'] = list(stats['languages'])
        
        return speaker_stats
    
    def _generate_language_stats(self, segments: List[ProcessedSegment]) -> Dict[str, Dict]:
        """Generate per-language statistics."""
        language_stats = {}
        
        for segment in segments:
            lang = segment.original_language
            
            if lang not in language_stats:
                language_stats[lang] = {
                    'speaking_time': 0.0,
                    'segment_count': 0,
                    'speakers': set()
                }
            
            stats = language_stats[lang]
            stats['speaking_time'] += segment.duration
            stats['segment_count'] += 1
            stats['speakers'].add(segment.speaker_id)
        
        # Convert sets to lists
        for lang, stats in language_stats.items():
            stats['speakers'] = list(stats['speakers'])
        
        return language_stats
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
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
    
    def _format_speaker_name(self, speaker_id: str) -> str:
        """Format speaker ID into a readable name."""
        if speaker_id.startswith("SPEAKER_"):
            number = speaker_id.replace("SPEAKER_", "")
            return f"Speaker {number}"
        return speaker_id.replace("_", " ").title()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience functions for easy usage
def create_processed_segment(start_time: float,
                           end_time: float,
                           speaker_id: str,
                           original_text: str,
                           original_language: str,
                           translated_text: str,
                           **kwargs) -> ProcessedSegment:
    """
    Convenience function to create a ProcessedSegment.
    
    Args:
        start_time (float): Segment start time
        end_time (float): Segment end time
        speaker_id (str): Speaker identifier
        original_text (str): Original transcribed text
        original_language (str): Original language code
        translated_text (str): Translated text
        **kwargs: Additional optional parameters
        
    Returns:
        ProcessedSegment: Created segment object
    """
    return ProcessedSegment(
        start_time=start_time,
        end_time=end_time,
        speaker_id=speaker_id,
        original_text=original_text,
        original_language=original_language,
        translated_text=translated_text,
        **kwargs
    )


def format_pipeline_output(diarization_segments,
                         transcription_segments, 
                         translation_results,
                         audio_filename: str = "audio_file",
                         audio_metadata: Optional[Dict] = None) -> Dict[str, str]:
    """
    Convenience function to format complete pipeline output.
    
    Args:
        diarization_segments: Speaker diarization results
        transcription_segments: Speech recognition results
        translation_results: Translation results
        audio_filename (str): Original audio filename
        audio_metadata (Dict, optional): Audio file metadata
        
    Returns:
        Dict[str, str]: All formatted outputs
    """
    # Combine all results into ProcessedSegment objects
    processed_segments = []
    
    # This is a simplified combination - in practice you'd need proper alignment
    for i, (diar_seg, trans_seg, trans_result) in enumerate(
        zip(diarization_segments, transcription_segments, translation_results)
    ):
        segment = ProcessedSegment(
            start_time=diar_seg.start_time,
            end_time=diar_seg.end_time,
            speaker_id=diar_seg.speaker_id,
            original_text=trans_seg.text,
            original_language=trans_seg.language,
            translated_text=trans_result.translated_text,
            confidence_diarization=diar_seg.confidence,
            confidence_transcription=trans_seg.confidence,
            confidence_translation=trans_result.confidence,
            word_timestamps=trans_seg.word_timestamps
        )
        processed_segments.append(segment)
    
    # Format all outputs
    formatter = OutputFormatter(audio_filename)
    return formatter.format_all_outputs(processed_segments, audio_metadata)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    def main():
        """Command line interface for testing output formatting."""
        parser = argparse.ArgumentParser(description="Audio Analysis Output Formatter")
        parser.add_argument("--demo", action="store_true",
                          help="Run with demo data")
        parser.add_argument("--format", choices=["json", "srt", "text", "csv", "timeline", "all"],
                          default="all", help="Output format to generate")
        parser.add_argument("--output-file", "-o",
                          help="Save output to file instead of printing")
        
        args = parser.parse_args()
        
        if args.demo:
            # Create demo data
            demo_segments = [
                ProcessedSegment(
                    start_time=0.0, end_time=3.5,
                    speaker_id="SPEAKER_00",
                    original_text="Hello, how are you today?",
                    original_language="en",
                    translated_text="Hello, how are you today?",
                    confidence_diarization=0.95,
                    confidence_transcription=0.92,
                    confidence_translation=1.0,
                    word_timestamps=[
                        {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.99},
                        {"word": "how", "start": 1.0, "end": 1.2, "confidence": 0.98},
                        {"word": "are", "start": 1.3, "end": 1.5, "confidence": 0.97},
                        {"word": "you", "start": 1.6, "end": 1.9, "confidence": 0.98},
                        {"word": "today", "start": 2.5, "end": 3.2, "confidence": 0.96}
                    ]
                ),
                ProcessedSegment(
                    start_time=4.0, end_time=7.8,
                    speaker_id="SPEAKER_01",
                    original_text="Bonjour, comment allez-vous?",
                    original_language="fr",
                    translated_text="Hello, how are you?",
                    confidence_diarization=0.87,
                    confidence_transcription=0.89,
                    confidence_translation=0.94
                ),
                ProcessedSegment(
                    start_time=8.5, end_time=12.1,
                    speaker_id="SPEAKER_00",
                    original_text="I'm doing well, thank you. What about you?",
                    original_language="en",
                    translated_text="I'm doing well, thank you. What about you?",
                    confidence_diarization=0.93,
                    confidence_transcription=0.95,
                    confidence_translation=1.0
                ),
                ProcessedSegment(
                    start_time=13.0, end_time=16.2,
                    speaker_id="SPEAKER_01",
                    original_text="Ça va très bien, merci beaucoup!",
                    original_language="fr",
                    translated_text="I'm doing very well, thank you very much!",
                    confidence_diarization=0.91,
                    confidence_transcription=0.88,
                    confidence_translation=0.92
                )
            ]
            
            demo_metadata = {
                "duration_seconds": 16.2,
                "sample_rate": 16000,
                "channels": 1
            }
            
            # Create formatter and generate output
            formatter = OutputFormatter("demo_conversation.wav")
            
            if args.format == "all":
                outputs = formatter.format_all_outputs(demo_segments, demo_metadata)
                
                if args.output_file:
                    # Save each format to separate files
                    base_name = args.output_file.rsplit('.', 1)[0]
                    for format_type, content in outputs.items():
                        filename = f"{base_name}.{format_type}"
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Saved {format_type} output to {filename}")
                else:
                    # Print all formats
                    for format_type, content in outputs.items():
                        print(f"\n{'='*20} {format_type.upper()} {'='*20}")
                        print(content)
            else:
                # Generate specific format
                if args.format == "json":
                    output = formatter.to_json(demo_segments, demo_metadata)
                elif args.format == "srt":
                    output = formatter.to_srt(demo_segments, use_translation=False)
                elif args.format == "text":
                    output = formatter.to_text(demo_segments, demo_metadata)
                elif args.format == "csv":
                    output = formatter.to_csv(demo_segments)
                elif args.format == "timeline":
                    output = formatter.to_timeline_json(demo_segments)
                
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        f.write(output)
                    print(f"Output saved to {args.output_file}")
                else:
                    print(output)
        
        else:
            print("Please use --demo flag to run with demo data, or integrate with your audio processing pipeline.")
    
    main() 