"""
Speaker Verification Module for PS-6 Requirements

This module extends beyond speaker diarization to include speaker identification
and verification capabilities using speaker embeddings and similarity matching.
"""

import numpy as np
import torch
import torchaudio
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class SpeakerVerifier:
    """
    Speaker verification system using speaker embeddings for identification
    and verification tasks beyond basic diarization.
    """
    
    def __init__(self, device: str = "cpu", cache_dir: str = "./model_cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.speaker_database = {}
        self.embedding_model = None
        self.similarity_threshold = 0.7  # Cosine similarity threshold for verification
        
        # Initialize the speaker verification model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the speaker embedding model."""
        try:
            # Try multiple advanced speaker embedding models for enhanced performance
            models_to_try = [
                "speechbrain/spkrec-ecapa-voxceleb",
                "speechbrain/spkrec-xvect-voxceleb",
                "microsoft/DialoGPT-medium",  # For conversational context
                "facebook/wav2vec2-base-960h"  # For robust feature extraction
            ]
            
            for model_name in models_to_try:
                try:
                    if "speechbrain" in model_name:
                        from speechbrain.pretrained import EncoderClassifier
                        self.embedding_model = EncoderClassifier.from_hparams(
                            source=model_name,
                            savedir=f"{self.cache_dir}/speechbrain_models/{model_name.split('/')[-1]}",
                            run_opts={"device": self.device}
                        )
                        self.model_type = "speechbrain"
                        logger.info(f"Loaded SpeechBrain model: {model_name}")
                        break
                        
                    elif "wav2vec2" in model_name:
                        from transformers import Wav2Vec2Model, Wav2Vec2Processor
                        self.embedding_model = Wav2Vec2Model.from_pretrained(model_name)
                        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                        self.model_type = "wav2vec2"
                        logger.info(f"Loaded Wav2Vec2 model: {model_name}")
                        break
                        
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue
            
            if self.embedding_model is None:
                # Fallback to pyannote
                try:
                    from pyannote.audio import Model
                    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
                    
                    self.embedding_model = PretrainedSpeakerEmbedding(
                        "speechbrain/spkrec-ecapa-voxceleb",
                        device=torch.device(self.device)
                    )
                    self.model_type = "pyannote"
                    logger.info("Loaded pyannote speaker embedding model")
                    
                except Exception as e:
                    logger.warning(f"Could not load any speaker embedding model: {e}")
                    logger.info("Falling back to basic speaker verification using diarization embeddings")
                    self.embedding_model = None
                    self.model_type = "basic"
            
        except Exception as e:
            logger.error(f"Error initializing speaker verification models: {e}")
            self.embedding_model = None
            self.model_type = "basic"
    
    def extract_speaker_embedding(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """
        Extract speaker embedding from audio segment using advanced models.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Speaker embedding vector
        """
        try:
            if self.embedding_model is not None and self.model_type != "basic":
                # Load and segment audio
                import librosa
                y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
                
                if self.model_type == "speechbrain":
                    # Use SpeechBrain models for enhanced performance
                    waveform = torch.from_numpy(y).unsqueeze(0)
                    embedding = self.embedding_model.encode_batch(waveform)
                    return embedding.squeeze().cpu().numpy()
                    
                elif self.model_type == "wav2vec2":
                    # Use Wav2Vec2 for robust feature extraction
                    inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        outputs = self.embedding_model(**inputs)
                        # Use mean pooling of last hidden states
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    return embedding.cpu().numpy()
                    
                elif self.model_type == "pyannote":
                    # Use pyannote's speaker embedding model
                    from pyannote.audio import Audio
                    audio = Audio(sample_rate=16000, mono=True)
                    waveform, sample_rate = audio.crop(audio_path, start_time, end_time)
                    embedding = self.embedding_model({"waveform": waveform, "sample_rate": sample_rate})
                    return embedding.cpu().numpy().flatten()
            
            else:
                # Fallback: Use enhanced basic features
                return self._extract_enhanced_features(audio_path, start_time, end_time)
                
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return np.zeros(512)  # Return zero vector as fallback
    
    def _extract_enhanced_features(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract enhanced audio features for advanced speaker verification."""
        try:
            import librosa
            
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
            
            # Enhanced feature extraction for advanced performance
            features = []
            
            # 1. MFCC features (13 coefficients + deltas + delta-deltas)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_deltas = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features.extend([
                np.mean(mfccs, axis=1),
                np.mean(mfcc_deltas, axis=1),
                np.mean(mfcc_delta2, axis=1)
            ])
            
            # 2. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features.extend([
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.mean(zero_crossing_rate)
            ])
            
            # 3. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.append(np.mean(chroma, axis=1))
            
            # 4. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.append(np.mean(tonnetz, axis=1))
            
            # 5. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.append(np.mean(contrast, axis=1))
            
            # 6. Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append([tempo])
            
            # 7. Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            features.append([np.mean(pitches), np.std(pitches)])
            
            # Combine all features
            combined_features = np.concatenate(features)
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(combined_features.reshape(-1, 1)).flatten()
            
            # Pad or truncate to fixed size
            if len(normalized_features) < 512:
                normalized_features = np.pad(normalized_features, (0, 512 - len(normalized_features)))
            else:
                normalized_features = normalized_features[:512]
                
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return self._extract_basic_features(audio_path, start_time, end_time)
    
    def _extract_basic_features(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract basic audio features as fallback embedding."""
        try:
            import librosa
            
            # Load audio segment
            y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            # Pad or truncate to fixed size
            if len(features) < 512:
                features = np.pad(features, (0, 512 - len(features)))
            else:
                features = features[:512]
                
            return features
            
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            return np.zeros(512)
    
    def enroll_speaker(self, speaker_id: str, audio_path: str, segments: List[Tuple[float, float]]) -> bool:
        """
        Enroll a speaker in the verification database.
        
        Args:
            speaker_id: Unique identifier for the speaker
            audio_path: Path to audio file
            segments: List of (start_time, end_time) tuples for speaker segments
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            embeddings = []
            
            for start_time, end_time in segments:
                embedding = self.extract_speaker_embedding(audio_path, start_time, end_time)
                embeddings.append(embedding)
            
            if embeddings:
                # Store multiple embeddings for robust verification
                self.speaker_database[speaker_id] = {
                    'embeddings': embeddings,
                    'mean_embedding': np.mean(embeddings, axis=0),
                    'audio_path': audio_path,
                    'enrollment_time': len(embeddings)
                }
                
                # Save to disk
                self._save_speaker_database()
                logger.info(f"Speaker {speaker_id} enrolled successfully with {len(embeddings)} segments")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error enrolling speaker {speaker_id}: {e}")
            return False
    
    def verify_speaker(self, speaker_id: str, audio_path: str, start_time: float, end_time: float) -> Dict:
        """
        Verify if an audio segment belongs to a known speaker using advanced methods.
        
        Args:
            speaker_id: Speaker to verify against
            audio_path: Path to audio file
            start_time: Start time of segment
            end_time: End time of segment
            
        Returns:
            Dictionary with verification results
        """
        try:
            if speaker_id not in self.speaker_database:
                return {
                    'verified': False,
                    'confidence': 0.0,
                    'error': f"Speaker {speaker_id} not found in database"
                }
            
            # Extract embedding from test segment
            test_embedding = self.extract_speaker_embedding(audio_path, start_time, end_time)
            
            # Get speaker's stored embeddings
            speaker_data = self.speaker_database[speaker_id]
            stored_embeddings = speaker_data['embeddings']
            mean_embedding = speaker_data['mean_embedding']
            
            # Advanced verification using multiple similarity metrics
            similarities = []
            euclidean_distances = []
            
            for stored_embedding in stored_embeddings:
                # Cosine similarity
                cos_sim = cosine_similarity([test_embedding], [stored_embedding])[0][0]
                similarities.append(cos_sim)
                
                # Euclidean distance (normalized)
                euclidean_dist = np.linalg.norm(test_embedding - stored_embedding)
                euclidean_distances.append(euclidean_dist)
            
            # Calculate multiple similarity metrics
            max_similarity = max(similarities)
            mean_similarity = np.mean(similarities)
            min_euclidean = min(euclidean_distances)
            mean_euclidean = np.mean(euclidean_distances)
            
            # Advanced confidence scoring using multiple metrics
            # Normalize euclidean distance to similarity (0-1 range)
            euclidean_similarity = 1 / (1 + mean_euclidean)
            
            # Weighted combination of multiple metrics
            confidence = (
                0.4 * max_similarity +           # Best cosine similarity
                0.3 * mean_similarity +          # Average cosine similarity
                0.2 * euclidean_similarity +     # Euclidean-based similarity
                0.1 * (1 - min_euclidean / (1 + min_euclidean))  # Min distance similarity
            )
            
            # Dynamic threshold based on enrollment quality
            dynamic_threshold = self.similarity_threshold
            if len(stored_embeddings) >= 5:
                dynamic_threshold *= 0.95  # Lower threshold for well-enrolled speakers
            elif len(stored_embeddings) < 3:
                dynamic_threshold *= 1.05  # Higher threshold for poorly enrolled speakers
            
            # Verification decision
            verified = confidence >= dynamic_threshold
            
            # Additional confidence factors
            enrollment_quality = min(len(stored_embeddings) / 10.0, 1.0)  # 0-1 scale
            final_confidence = confidence * (0.8 + 0.2 * enrollment_quality)
            
            return {
                'verified': verified,
                'confidence': float(final_confidence),
                'raw_confidence': float(confidence),
                'max_similarity': float(max_similarity),
                'mean_similarity': float(mean_similarity),
                'euclidean_similarity': float(euclidean_similarity),
                'threshold': float(dynamic_threshold),
                'enrollment_segments': len(stored_embeddings),
                'enrollment_quality': float(enrollment_quality),
                'verification_method': self.model_type
            }
            
        except Exception as e:
            logger.error(f"Error verifying speaker {speaker_id}: {e}")
            return {
                'verified': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def identify_speaker(self, audio_path: str, start_time: float, end_time: float) -> Dict:
        """
        Identify the most likely speaker from the enrolled database.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time of segment
            end_time: End time of segment
            
        Returns:
            Dictionary with identification results
        """
        try:
            if not self.speaker_database:
                return {
                    'identified_speaker': None,
                    'confidence': 0.0,
                    'error': "No speakers enrolled in database"
                }
            
            # Extract embedding from test segment
            test_embedding = self.extract_speaker_embedding(audio_path, start_time, end_time)
            
            best_speaker = None
            best_confidence = 0.0
            all_scores = {}
            
            # Compare against all enrolled speakers
            for speaker_id, speaker_data in self.speaker_database.items():
                stored_embeddings = speaker_data['embeddings']
                
                similarities = []
                for stored_embedding in stored_embeddings:
                    similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]
                    similarities.append(similarity)
                
                confidence = np.mean(similarities)
                all_scores[speaker_id] = confidence
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_speaker = speaker_id
            
            return {
                'identified_speaker': best_speaker,
                'confidence': float(best_confidence),
                'all_scores': all_scores,
                'threshold': self.similarity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error identifying speaker: {e}")
            return {
                'identified_speaker': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _save_speaker_database(self):
        """Save speaker database to disk."""
        try:
            db_path = self.cache_dir / "speaker_database.pkl"
            self.cache_dir.mkdir(exist_ok=True)
            
            with open(db_path, 'wb') as f:
                pickle.dump(self.speaker_database, f)
                
        except Exception as e:
            logger.error(f"Error saving speaker database: {e}")
    
    def _load_speaker_database(self):
        """Load speaker database from disk."""
        try:
            db_path = self.cache_dir / "speaker_database.pkl"
            if db_path.exists():
                with open(db_path, 'rb') as f:
                    self.speaker_database = pickle.load(f)
                logger.info(f"Loaded speaker database with {len(self.speaker_database)} speakers")
                
        except Exception as e:
            logger.error(f"Error loading speaker database: {e}")
            self.speaker_database = {}
    
    def get_speaker_statistics(self) -> Dict:
        """Get statistics about enrolled speakers."""
        if not self.speaker_database:
            return {'total_speakers': 0, 'speakers': []}
        
        speakers_info = []
        for speaker_id, data in self.speaker_database.items():
            speakers_info.append({
                'speaker_id': speaker_id,
                'enrollment_segments': data['enrollment_time'],
                'audio_path': data['audio_path']
            })
        
        return {
            'total_speakers': len(self.speaker_database),
            'speakers': speakers_info
        }
    
    def clear_database(self):
        """Clear all enrolled speakers."""
        self.speaker_database = {}
        self._save_speaker_database()
        logger.info("Speaker database cleared")
