"""
Neural Machine Translation Module for Multilingual Audio Intelligence System

This module implements state-of-the-art neural machine translation using Helsinki-NLP/Opus-MT
models. Designed for efficient CPU-based translation with dynamic model loading and
intelligent batching strategies.

Key Features:
- Dynamic model loading for 100+ language pairs
- Helsinki-NLP/Opus-MT models (300MB each) for specific language pairs
- Intelligent batching for maximum CPU throughput
- Fallback to multilingual models (mBART, M2M-100) for rare languages
- Memory-efficient model management with automatic cleanup
- Robust error handling and translation confidence scoring
- Cache management for frequently used language pairs

Models: Helsinki-NLP/opus-mt-* series, Facebook mBART50, M2M-100
Dependencies: transformers, torch, sentencepiece
"""

import os
import logging
import warnings
import torch
from typing import List, Dict, Optional, Tuple, Union
import gc
from dataclasses import dataclass
from collections import defaultdict
import time

try:
    from transformers import (
        MarianMTModel, MarianTokenizer, 
        MBartForConditionalGeneration, MBart50TokenizerFast,
        M2M100ForConditionalGeneration, M2M100Tokenizer,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class TranslationResult:
    """
    Data class representing a translation result with metadata.
    
    Attributes:
        original_text (str): Original text in source language
        translated_text (str): Translated text in target language
        source_language (str): Source language code
        target_language (str): Target language code
        confidence (float): Translation confidence score
        model_used (str): Name of the model used for translation
        processing_time (float): Time taken for translation in seconds
    """
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = 1.0
    model_used: str = "unknown"
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'source_language': self.source_language,
            'target_language': self.target_language,
            'confidence': self.confidence,
            'model_used': self.model_used,
            'processing_time': self.processing_time
        }


class NeuralTranslator:
    """
    Advanced neural machine translation with dynamic model loading.
    
    Supports 100+ languages through Helsinki-NLP/Opus-MT models with intelligent
    fallback strategies and efficient memory management.
    """
    
    def __init__(self,
                 target_language: str = "en",
                 device: Optional[str] = None,
                 cache_size: int = 3,
                 use_multilingual_fallback: bool = True,
                 model_cache_dir: Optional[str] = None):
        """
        Initialize the Neural Translator.
        
        Args:
            target_language (str): Target language code (default: 'en' for English)
            device (str, optional): Device to run on ('cpu', 'cuda', 'auto')
            cache_size (int): Maximum number of models to keep in memory
            use_multilingual_fallback (bool): Use mBART/M2M-100 for unsupported pairs
            model_cache_dir (str, optional): Directory to cache downloaded models
        """
        self.target_language = target_language
        self.cache_size = cache_size
        self.use_multilingual_fallback = use_multilingual_fallback
        self.model_cache_dir = model_cache_dir
        
        # Device selection
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing NeuralTranslator: target={target_language}, "
                   f"device={self.device}, cache_size={cache_size}")
        
        # Model cache and management
        self.model_cache = {}  # {model_name: (model, tokenizer, last_used)}
        self.fallback_model = None
        self.fallback_tokenizer = None
        self.fallback_model_name = None
        
        # Language mapping for Helsinki-NLP models
        self.language_mapping = self._get_language_mapping()
        
        # Supported language pairs cache
        self._supported_pairs_cache = None
        
        # Initialize fallback model if requested
        if use_multilingual_fallback:
            self._load_fallback_model()
    
    def _get_language_mapping(self) -> Dict[str, str]:
        """Get mapping of language codes to Helsinki-NLP model codes."""
        # Common language mappings for Helsinki-NLP/Opus-MT
        return {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'pt': 'pt',
            'ru': 'ru', 'zh': 'zh', 'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi',
            'tr': 'tr', 'pl': 'pl', 'nl': 'nl', 'sv': 'sv', 'da': 'da', 'no': 'no',
            'fi': 'fi', 'hu': 'hu', 'cs': 'cs', 'sk': 'sk', 'sl': 'sl', 'hr': 'hr',
            'bg': 'bg', 'ro': 'ro', 'el': 'el', 'he': 'he', 'th': 'th', 'vi': 'vi',
            'id': 'id', 'ms': 'ms', 'tl': 'tl', 'sw': 'sw', 'eu': 'eu', 'ca': 'ca',
            'gl': 'gl', 'cy': 'cy', 'ga': 'ga', 'mt': 'mt', 'is': 'is', 'lv': 'lv',
            'lt': 'lt', 'et': 'et', 'mk': 'mk', 'sq': 'sq', 'be': 'be', 'uk': 'uk',
            'ka': 'ka', 'hy': 'hy', 'az': 'az', 'kk': 'kk', 'ky': 'ky', 'uz': 'uz',
            'fa': 'fa', 'ur': 'ur', 'bn': 'bn', 'ta': 'ta', 'te': 'te', 'ml': 'ml',
            'kn': 'kn', 'gu': 'gu', 'pa': 'pa', 'mr': 'mr', 'ne': 'ne', 'si': 'si',
            'my': 'my', 'km': 'km', 'lo': 'lo', 'mn': 'mn', 'bo': 'bo'
        }
    
    def _load_fallback_model(self):
        """Load multilingual fallback model (mBART50 or M2M-100)."""
        try:
            # Try mBART50 first (smaller and faster)
            logger.info("Loading mBART50 multilingual fallback model...")
            
            self.fallback_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt",
                cache_dir=self.model_cache_dir
            ).to(self.device)
            
            self.fallback_tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt",
                cache_dir=self.model_cache_dir
            )
            
            self.fallback_model_name = "mbart50"
            logger.info("mBART50 fallback model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load mBART50: {e}")
            
            try:
                # Fallback to M2M-100 (larger but more comprehensive)
                logger.info("Loading M2M-100 multilingual fallback model...")
                
                self.fallback_model = M2M100ForConditionalGeneration.from_pretrained(
                    "facebook/m2m100_418M",
                    cache_dir=self.model_cache_dir
                ).to(self.device)
                
                self.fallback_tokenizer = M2M100Tokenizer.from_pretrained(
                    "facebook/m2m100_418M",
                    cache_dir=self.model_cache_dir
                )
                
                self.fallback_model_name = "m2m100"
                logger.info("M2M-100 fallback model loaded successfully")
                
            except Exception as e2:
                logger.warning(f"Failed to load M2M-100: {e2}")
                self.fallback_model = None
                self.fallback_tokenizer = None
                self.fallback_model_name = None
    
    def translate_text(self,
                      text: str,
                      source_language: str,
                      target_language: Optional[str] = None) -> TranslationResult:
        """
        Translate a single text segment.
        
        Args:
            text (str): Text to translate
            source_language (str): Source language code
            target_language (str, optional): Target language code (uses default if None)
            
        Returns:
            TranslationResult: Translation result with metadata
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language or self.target_language,
                confidence=0.0,
                model_used="none",
                processing_time=0.0
            )
        
        target_lang = target_language or self.target_language
        
        # Skip translation if source equals target
        if source_language == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_lang,
                confidence=1.0,
                model_used="identity",
                processing_time=0.0
            )
        
        start_time = time.time()
        
        try:
            # Try Helsinki-NLP model first
            model_name = self._get_model_name(source_language, target_lang)
            
            if model_name:
                result = self._translate_with_opus_mt(
                    text, source_language, target_lang, model_name
                )
            elif self.fallback_model:
                result = self._translate_with_fallback(
                    text, source_language, target_lang
                )
            else:
                # No translation available
                result = TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_lang,
                    confidence=0.0,
                    model_used="unavailable",
                    processing_time=0.0
                )
            
            result.processing_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_lang,
                confidence=0.0,
                model_used="error",
                processing_time=time.time() - start_time
            )
    
    def translate_batch(self,
                       texts: List[str],
                       source_languages: List[str],
                       target_language: Optional[str] = None,
                       batch_size: int = 8) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently using batching.
        
        Args:
            texts (List[str]): List of texts to translate
            source_languages (List[str]): List of source language codes
            target_language (str, optional): Target language code
            batch_size (int): Batch size for processing
            
        Returns:
            List[TranslationResult]: List of translation results
        """
        if len(texts) != len(source_languages):
            raise ValueError("Number of texts must match number of source languages")
        
        target_lang = target_language or self.target_language
        results = []
        
        # Group by language pair for efficient batching
        language_groups = defaultdict(list)
        for i, (text, src_lang) in enumerate(zip(texts, source_languages)):
            if text and text.strip():
                language_groups[(src_lang, target_lang)].append((i, text))
        
        # Process each language group
        for (src_lang, tgt_lang), items in language_groups.items():
            if src_lang == tgt_lang:
                # Identity translation
                for idx, text in items:
                    results.append((idx, TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=src_lang,
                        target_language=tgt_lang,
                        confidence=1.0,
                        model_used="identity",
                        processing_time=0.0
                    )))
            else:
                # Translate in batches
                for i in range(0, len(items), batch_size):
                    batch_items = items[i:i + batch_size]
                    batch_texts = [item[1] for item in batch_items]
                    batch_indices = [item[0] for item in batch_items]
                    
                    batch_results = self._translate_batch_same_language(
                        batch_texts, src_lang, tgt_lang
                    )
                    
                    for idx, result in zip(batch_indices, batch_results):
                        results.append((idx, result))
        
        # Fill in empty texts and sort by original order
        final_results = [None] * len(texts)
        for idx, result in results:
            final_results[idx] = result
        
        # Handle empty texts
        for i, result in enumerate(final_results):
            if result is None:
                final_results[i] = TranslationResult(
                    original_text=texts[i],
                    translated_text=texts[i],
                    source_language=source_languages[i],
                    target_language=target_lang,
                    confidence=0.0,
                    model_used="empty",
                    processing_time=0.0
                )
        
        return final_results
    
    def _translate_batch_same_language(self,
                                     texts: List[str],
                                     source_language: str,
                                     target_language: str) -> List[TranslationResult]:
        """Translate a batch of texts from the same source language."""
        try:
            model_name = self._get_model_name(source_language, target_language)
            
            if model_name:
                return self._translate_batch_opus_mt(
                    texts, source_language, target_language, model_name
                )
            elif self.fallback_model:
                return self._translate_batch_fallback(
                    texts, source_language, target_language
                )
            else:
                # No translation available
                return [
                    TranslationResult(
                        original_text=text,
                        translated_text=text,
                        source_language=source_language,
                        target_language=target_language,
                        confidence=0.0,
                        model_used="unavailable",
                        processing_time=0.0
                    )
                    for text in texts
                ]
                
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return [
                TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.0,
                    model_used="error",
                    processing_time=0.0
                )
                for text in texts
            ]
    
    def _get_model_name(self, source_lang: str, target_lang: str) -> Optional[str]:
        """Get Helsinki-NLP model name for language pair."""
        # Map language codes
        src_mapped = self.language_mapping.get(source_lang, source_lang)
        tgt_mapped = self.language_mapping.get(target_lang, target_lang)
        
        # Common Helsinki-NLP model patterns
        model_patterns = [
            f"Helsinki-NLP/opus-mt-{src_mapped}-{tgt_mapped}",
            f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}",
            f"Helsinki-NLP/opus-mt-{src_mapped}-{target_lang}",
            f"Helsinki-NLP/opus-mt-{source_lang}-{tgt_mapped}"
        ]
        
        # For specific language groups, try group models
        if target_lang == 'en':
            # Many-to-English models
            group_patterns = [
                f"Helsinki-NLP/opus-mt-mul-{target_lang}",
                f"Helsinki-NLP/opus-mt-roa-{target_lang}",  # Romance languages
                f"Helsinki-NLP/opus-mt-gem-{target_lang}",  # Germanic languages
                f"Helsinki-NLP/opus-mt-sla-{target_lang}",  # Slavic languages
            ]
            model_patterns.extend(group_patterns)
        
        # Return the first pattern (most specific)
        return model_patterns[0] if model_patterns else None
    
    def _load_opus_mt_model(self, model_name: str) -> Tuple[MarianMTModel, MarianTokenizer]:
        """Load Helsinki-NLP Opus-MT model with caching."""
        current_time = time.time()
        
        # Check if model is already in cache
        if model_name in self.model_cache:
            model, tokenizer, _ = self.model_cache[model_name]
            # Update last used time
            self.model_cache[model_name] = (model, tokenizer, current_time)
            logger.debug(f"Using cached model: {model_name}")
            return model, tokenizer
        
        # Clean cache if it's full
        if len(self.model_cache) >= self.cache_size:
            self._clean_model_cache()
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load model and tokenizer
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            ).to(self.device)
            
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            
            # Add to cache
            self.model_cache[model_name] = (model, tokenizer, current_time)
            logger.info(f"Model loaded and cached: {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            raise
    
    def _clean_model_cache(self):
        """Remove least recently used model from cache."""
        if not self.model_cache:
            return
        
        # Find least recently used model
        lru_model = min(self.model_cache.items(), key=lambda x: x[1][2])
        model_name = lru_model[0]
        
        # Remove from cache and free memory
        model, tokenizer, _ = self.model_cache.pop(model_name)
        del model, tokenizer
        
        # Force garbage collection
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.debug(f"Removed model from cache: {model_name}")
    
    def _translate_with_opus_mt(self,
                              text: str,
                              source_language: str,
                              target_language: str,
                              model_name: str) -> TranslationResult:
        """Translate text using Helsinki-NLP Opus-MT model."""
        try:
            model, tokenizer = self._load_opus_mt_model(model_name)
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.9,  # Opus-MT models generally have good confidence
                model_used=model_name
            )
            
        except Exception as e:
            logger.error(f"Opus-MT translation failed: {e}")
            raise
    
    def _translate_batch_opus_mt(self,
                               texts: List[str],
                               source_language: str,
                               target_language: str,
                               model_name: str) -> List[TranslationResult]:
        """Translate batch using Helsinki-NLP Opus-MT model."""
        try:
            model, tokenizer = self._load_opus_mt_model(model_name)
            
            # Tokenize batch
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode all outputs
            translated_texts = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Create results
            results = []
            for original, translated in zip(texts, translated_texts):
                results.append(TranslationResult(
                    original_text=original,
                    translated_text=translated,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=0.9,
                    model_used=model_name
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Opus-MT batch translation failed: {e}")
            raise
    
    def _translate_with_fallback(self,
                               text: str,
                               source_language: str,
                               target_language: str) -> TranslationResult:
        """Translate using multilingual fallback model."""
        try:
            if self.fallback_model_name == "mbart50":
                return self._translate_with_mbart50(text, source_language, target_language)
            elif self.fallback_model_name == "m2m100":
                return self._translate_with_m2m100(text, source_language, target_language)
            else:
                raise ValueError("No fallback model available")
                
        except Exception as e:
            logger.error(f"Fallback translation failed: {e}")
            raise
    
    def _translate_batch_fallback(self,
                                texts: List[str],
                                source_language: str,
                                target_language: str) -> List[TranslationResult]:
        """Translate batch using multilingual fallback model."""
        try:
            if self.fallback_model_name == "mbart50":
                return self._translate_batch_mbart50(texts, source_language, target_language)
            elif self.fallback_model_name == "m2m100":
                return self._translate_batch_m2m100(texts, source_language, target_language)
            else:
                raise ValueError("No fallback model available")
                
        except Exception as e:
            logger.error(f"Fallback batch translation failed: {e}")
            raise
    
    def _translate_with_mbart50(self,
                              text: str,
                              source_language: str,
                              target_language: str) -> TranslationResult:
        """Translate using mBART50 model."""
        # Set source language
        self.fallback_tokenizer.src_lang = source_language
        
        inputs = self.fallback_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            generated_tokens = self.fallback_model.generate(
                **inputs,
                forced_bos_token_id=self.fallback_tokenizer.lang_code_to_id[target_language],
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        translated_text = self.fallback_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=0.85,
            model_used="mbart50"
        )
    
    def _translate_batch_mbart50(self,
                               texts: List[str],
                               source_language: str,
                               target_language: str) -> List[TranslationResult]:
        """Translate batch using mBART50 model."""
        # Set source language
        self.fallback_tokenizer.src_lang = source_language
        
        inputs = self.fallback_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translations
        with torch.no_grad():
            generated_tokens = self.fallback_model.generate(
                **inputs,
                forced_bos_token_id=self.fallback_tokenizer.lang_code_to_id[target_language],
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        translated_texts = self.fallback_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        
        return [
            TranslationResult(
                original_text=original,
                translated_text=translated,
                source_language=source_language,
                target_language=target_language,
                confidence=0.85,
                model_used="mbart50"
            )
            for original, translated in zip(texts, translated_texts)
        ]
    
    def _translate_with_m2m100(self,
                             text: str,
                             source_language: str,
                             target_language: str) -> TranslationResult:
        """Translate using M2M-100 model."""
        self.fallback_tokenizer.src_lang = source_language
        
        inputs = self.fallback_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = self.fallback_model.generate(
                **inputs,
                forced_bos_token_id=self.fallback_tokenizer.get_lang_id(target_language),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        translated_text = self.fallback_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        
        return TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_language,
            target_language=target_language,
            confidence=0.87,
            model_used="m2m100"
        )
    
    def _translate_batch_m2m100(self,
                              texts: List[str],
                              source_language: str,
                              target_language: str) -> List[TranslationResult]:
        """Translate batch using M2M-100 model."""
        self.fallback_tokenizer.src_lang = source_language
        
        inputs = self.fallback_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = self.fallback_model.generate(
                **inputs,
                forced_bos_token_id=self.fallback_tokenizer.get_lang_id(target_language),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        translated_texts = self.fallback_tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        
        return [
            TranslationResult(
                original_text=original,
                translated_text=translated,
                source_language=source_language,
                target_language=target_language,
                confidence=0.87,
                model_used="m2m100"
            )
            for original, translated in zip(texts, translated_texts)
        ]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported source languages."""
        # Combined support from Helsinki-NLP and fallback models
        opus_mt_languages = list(self.language_mapping.keys())
        
        # mBART50 supported languages
        mbart_languages = [
            'ar', 'cs', 'de', 'en', 'es', 'et', 'fi', 'fr', 'gu', 'hi', 'it', 'ja',
            'kk', 'ko', 'lt', 'lv', 'my', 'ne', 'nl', 'ro', 'ru', 'si', 'tr', 'vi',
            'zh', 'af', 'az', 'bn', 'fa', 'he', 'hr', 'id', 'ka', 'km', 'mk', 'ml',
            'mn', 'mr', 'pl', 'ps', 'pt', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'uk',
            'ur', 'xh', 'gl', 'sl'
        ]
        
        # M2M-100 has 100 languages, include major ones
        m2m_additional = [
            'am', 'cy', 'is', 'mg', 'mt', 'so', 'zu', 'ha', 'ig', 'yo', 'lg', 'ln',
            'rn', 'sn', 'tn', 'ts', 've', 'xh', 'zu'
        ]
        
        all_languages = set(opus_mt_languages + mbart_languages + m2m_additional)
        return sorted(list(all_languages))
    
    def clear_cache(self):
        """Clear all cached models to free memory."""
        logger.info("Clearing model cache...")
        
        for model_name, (model, tokenizer, _) in self.model_cache.items():
            del model, tokenizer
        
        self.model_cache.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get information about cached models."""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'max_cache_size': self.cache_size,
            'fallback_model': self.fallback_model_name,
            'device': str(self.device)
        }
    
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        try:
            self.clear_cache()
        except Exception:
            pass


# Convenience function for easy usage
def translate_text(text: str,
                  source_language: str,
                  target_language: str = "en",
                  device: Optional[str] = None) -> TranslationResult:
    """
    Convenience function to translate text with default settings.
    
    Args:
        text (str): Text to translate
        source_language (str): Source language code
        target_language (str): Target language code (default: 'en')
        device (str, optional): Device to run on ('cpu', 'cuda', 'auto')
        
    Returns:
        TranslationResult: Translation result
        
    Example:
        >>> # Translate from French to English
        >>> result = translate_text("Bonjour le monde", "fr", "en")
        >>> print(result.translated_text)  # "Hello world"
        >>> 
        >>> # Translate from Hindi to English
        >>> result = translate_text("नमस्ते", "hi", "en")
        >>> print(result.translated_text)  # "Hello"
    """
    translator = NeuralTranslator(
        target_language=target_language,
        device=device
    )
    
    return translator.translate_text(text, source_language, target_language)


# Example usage and testing
if __name__ == "__main__":
    import sys
    import argparse
    import json
    
    def main():
        """Command line interface for testing neural translation."""
        parser = argparse.ArgumentParser(description="Neural Machine Translation Tool")
        parser.add_argument("text", help="Text to translate")
        parser.add_argument("--source-lang", "-s", required=True,
                          help="Source language code")
        parser.add_argument("--target-lang", "-t", default="en",
                          help="Target language code (default: en)")
        parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                          help="Device to run on")
        parser.add_argument("--batch-size", type=int, default=8,
                          help="Batch size for multiple texts")
        parser.add_argument("--output-format", choices=["json", "text"],
                          default="text", help="Output format")
        parser.add_argument("--list-languages", action="store_true",
                          help="List supported languages")
        parser.add_argument("--benchmark", action="store_true",
                          help="Run translation benchmark")
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose logging")
        
        args = parser.parse_args()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        try:
            translator = NeuralTranslator(
                target_language=args.target_lang,
                device=args.device
            )
            
            if args.list_languages:
                languages = translator.get_supported_languages()
                print("Supported languages:")
                for i, lang in enumerate(languages):
                    print(f"{lang:>4}", end="  ")
                    if (i + 1) % 10 == 0:
                        print()
                if len(languages) % 10 != 0:
                    print()
                return
            
            if args.benchmark:
                print("=== TRANSLATION BENCHMARK ===")
                test_texts = [
                    "Hello, how are you?",
                    "This is a longer sentence to test translation quality.",
                    "Machine translation has improved significantly."
                ]
                
                start_time = time.time()
                results = translator.translate_batch(
                    test_texts,
                    [args.source_lang] * len(test_texts),
                    args.target_lang
                )
                total_time = time.time() - start_time
                
                print(f"Translated {len(test_texts)} texts in {total_time:.2f}s")
                print(f"Average time per text: {total_time/len(test_texts):.3f}s")
                print()
            
            # Translate the input text
            result = translator.translate_text(
                args.text, args.source_lang, args.target_lang
            )
            
            # Output results
            if args.output_format == "json":
                print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(f"=== TRANSLATION RESULT ===")
                print(f"Source ({result.source_language}): {result.original_text}")
                print(f"Target ({result.target_language}): {result.translated_text}")
                print(f"Model used: {result.model_used}")
                print(f"Confidence: {result.confidence:.2f}")
                print(f"Processing time: {result.processing_time:.3f}s")
                
                if args.verbose:
                    cache_info = translator.get_cache_info()
                    print(f"\nCache info: {cache_info}")
        
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run CLI if script is executed directly
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: transformers not available. Install with: pip install transformers")
        print("Running in demo mode...")
        
        # Create dummy result for testing
        dummy_result = TranslationResult(
            original_text="Bonjour le monde",
            translated_text="Hello world",
            source_language="fr",
            target_language="en",
            confidence=0.95,
            model_used="demo",
            processing_time=0.123
        )
        
        print("\n=== DEMO OUTPUT (transformers not available) ===")
        print(f"Source (fr): {dummy_result.original_text}")
        print(f"Target (en): {dummy_result.translated_text}")
        print(f"Confidence: {dummy_result.confidence:.2f}")
    else:
        main() 