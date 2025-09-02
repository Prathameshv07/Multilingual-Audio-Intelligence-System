"""
Quality Control Module for Audio Intelligence System

This module implements quality checks and model selection strategies
to ensure the system only demonstrates its best capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class QualityController:
    """
    Controls quality of transcription and translation to avoid
    misleading results in demonstrations.
    """
    
    def __init__(self):
        # Languages where we have good model performance
        self.reliable_languages = {
            'hi': {'name': 'Hindi', 'opus_mt': True, 'quality': 'high'},
            'ja': {'name': 'Japanese', 'opus_mt': True, 'quality': 'high'},
            'fr': {'name': 'French', 'opus_mt': True, 'quality': 'high'},
            'en': {'name': 'English', 'opus_mt': True, 'quality': 'high'},
            'ur': {'name': 'Urdu', 'opus_mt': True, 'quality': 'medium'},
            'bn': {'name': 'Bengali', 'opus_mt': True, 'quality': 'medium'},
        }
        
        # Patterns that indicate poor transcription quality
        self.poor_quality_patterns = [
            r'^(.+?)\1{4,}',  # Repetitive patterns (word repeated 4+ times)
            r'^(तो\s*){10,}',  # Specific Hindi repetition issue
            r'^(.{1,3}\s*){20,}',  # Very short repeated phrases
        ]
    
    def validate_language_detection(self, text: str, detected_lang: str) -> Tuple[str, float]:
        """
        Validate language detection and return corrected language with confidence.
        
        Returns:
            Tuple[str, float]: (corrected_language, confidence)
        """
        # Clean text for analysis
        clean_text = text.strip()
        
        # Script-based detection for Indian languages
        devanagari_chars = sum(1 for char in clean_text if '\u0900' <= char <= '\u097F')
        arabic_chars = sum(1 for char in clean_text if '\u0600' <= char <= '\u06FF')
        latin_chars = sum(1 for char in clean_text if char.isascii() and char.isalpha())
        
        total_chars = len([c for c in clean_text if c.isalpha()])
        
        if total_chars == 0:
            return detected_lang, 0.1
        
        # Calculate script ratios
        devanagari_ratio = devanagari_chars / total_chars
        arabic_ratio = arabic_chars / total_chars
        latin_ratio = latin_chars / total_chars
        
        # High confidence script-based detection
        if devanagari_ratio > 0.8:
            return 'hi', 0.95
        elif arabic_ratio > 0.8:
            return 'ur', 0.9
        elif latin_ratio > 0.9:
            # Could be English, French, or romanized text
            if detected_lang in ['en', 'fr']:
                return detected_lang, 0.8
            return 'en', 0.7
        
        # Medium confidence corrections
        if devanagari_ratio > 0.5:
            return 'hi', 0.7
        elif arabic_ratio > 0.5:
            return 'ur', 0.7
        
        # If current detection is unreliable, default to Hindi for Indian audio
        if detected_lang in ['zh', 'th', 'ko'] and devanagari_ratio > 0.2:
            return 'hi', 0.6
        
        return detected_lang, 0.5
    
    def assess_transcription_quality(self, text: str) -> Dict[str, any]:
        """
        Assess the quality of transcribed text.
        
        Returns:
            Dict with quality assessment
        """
        clean_text = text.strip()
        words = clean_text.split()
        
        assessment = {
            'text': clean_text,
            'quality_score': 1.0,
            'issues': [],
            'recommendation': 'accept'
        }
        
        # Check text length
        if len(clean_text) < 5:
            assessment['quality_score'] *= 0.3
            assessment['issues'].append('very_short')
        
        if len(words) == 0:
            assessment['quality_score'] = 0.0
            assessment['issues'].append('empty')
            assessment['recommendation'] = 'reject'
            return assessment
        
        # Check for repetition
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words)
        
        if repetition_ratio < 0.3:
            assessment['quality_score'] *= 0.2
            assessment['issues'].append('highly_repetitive')
            assessment['recommendation'] = 'filter'
        elif repetition_ratio < 0.5:
            assessment['quality_score'] *= 0.6
            assessment['issues'].append('repetitive')
        
        # Check for specific poor quality patterns
        for pattern in self.poor_quality_patterns:
            if re.match(pattern, clean_text):
                assessment['quality_score'] *= 0.1
                assessment['issues'].append('pattern_match')
                assessment['recommendation'] = 'reject'
                break
        
        # Check for garbled text (too many non-word characters)
        alpha_ratio = len([c for c in clean_text if c.isalpha()]) / max(1, len(clean_text))
        if alpha_ratio < 0.5:
            assessment['quality_score'] *= 0.4
            assessment['issues'].append('garbled')
        
        # Final recommendation
        if assessment['quality_score'] < 0.2:
            assessment['recommendation'] = 'reject'
        elif assessment['quality_score'] < 0.5:
            assessment['recommendation'] = 'filter'
        
        return assessment
    
    def should_process_language(self, language: str) -> bool:
        """
        Determine if we should process this language based on our capabilities.
        """
        return language in self.reliable_languages
    
    def get_best_translation_strategy(self, source_lang: str, target_lang: str) -> Dict[str, any]:
        """
        Get the best translation strategy for the language pair.
        """
        strategy = {
            'method': 'hybrid',
            'confidence': 0.5,
            'explanation': 'Standard hybrid approach'
        }
        
        if source_lang not in self.reliable_languages:
            strategy['method'] = 'google_only'
            strategy['confidence'] = 0.6
            strategy['explanation'] = f'Language {source_lang} not in reliable set, using Google API'
        elif self.reliable_languages[source_lang]['quality'] == 'high':
            strategy['confidence'] = 0.9
            strategy['explanation'] = f'High quality support for {source_lang}'
        
        return strategy
    
    def filter_results_for_demo(self, segments: List) -> List:
        """
        Filter results to show only high-quality segments for demo purposes.
        """
        filtered_segments = []
        
        for segment in segments:
            # Assess transcription quality
            quality = self.assess_transcription_quality(segment.original_text)
            
            if quality['recommendation'] == 'accept':
                filtered_segments.append(segment)
            elif quality['recommendation'] == 'filter':
                # Keep but mark as filtered
                segment.original_text = f"[Filtered] {segment.original_text}"
                segment.confidence_transcription *= 0.5
                filtered_segments.append(segment)
            # Skip 'reject' segments entirely
        
        logger.info(f"Quality filter: {len(segments)} → {len(filtered_segments)} segments")
        return filtered_segments

# Global instance
quality_controller = QualityController()


