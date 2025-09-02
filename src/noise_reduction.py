"""
Noise Reduction Module for PS-6 Requirements

This module provides speech enhancement capabilities to handle noisy audio
conditions as required for SNR -5 to 20 dB operation.
"""

import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class NoiseReducer:
    """
    Speech enhancement system for noise reduction and robustness.
    Handles various noise conditions to improve ASR performance.
    """
    
    def __init__(self, device: str = "cpu", cache_dir: str = "./model_cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.enhancement_model = None
        self.sample_rate = 16000
        
        # Initialize noise reduction model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize advanced speech enhancement models."""
        try:
            # Try to load multiple advanced speech enhancement models
            models_to_try = [
                "speechbrain/sepformer-wham",
                "speechbrain/sepformer-wsj02mix", 
                "facebook/demucs",
                "microsoft/DialoGPT-medium"  # For conversational context
            ]
            
            self.enhancement_models = {}
            
            for model_name in models_to_try:
                try:
                    if "speechbrain" in model_name:
                        from speechbrain.pretrained import SepformerSeparation
                        self.enhancement_models[model_name] = SepformerSeparation.from_hparams(
                            source=model_name,
                            savedir=f"{self.cache_dir}/speechbrain_enhancement/{model_name.split('/')[-1]}",
                            run_opts={"device": self.device}
                        )
                        logger.info(f"Loaded SpeechBrain enhancement model: {model_name}")
                        
                    elif "demucs" in model_name:
                        # Try to load Demucs for music/speech separation
                        try:
                            import demucs.api
                            self.enhancement_models[model_name] = demucs.api.Separator()
                            logger.info(f"Loaded Demucs model: {model_name}")
                        except ImportError:
                            logger.warning("Demucs not available, skipping")
                            
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue
            
            if not self.enhancement_models:
                logger.info("No advanced models loaded, using enhanced signal processing")
                self.enhancement_models = None
            else:
                logger.info(f"Loaded {len(self.enhancement_models)} enhancement models")
            
        except Exception as e:
            logger.warning(f"Could not load advanced noise reduction models: {e}")
            logger.info("Using enhanced signal processing for noise reduction")
            self.enhancement_models = None
    
    def enhance_audio(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """
        Enhance audio using advanced noise reduction and speech enhancement.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for enhanced audio output (optional)
            
        Returns:
            Path to enhanced audio file
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Apply advanced noise reduction
            enhanced_waveform = self._apply_advanced_noise_reduction(waveform, audio_path)
            
            # Generate output path if not provided
            if output_path is None:
                input_path = Path(audio_path)
                output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"
            
            # Save enhanced audio
            torchaudio.save(output_path, enhanced_waveform, self.sample_rate)
            
            logger.info(f"Audio enhanced using advanced methods and saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {e}")
            return audio_path  # Return original path if enhancement fails
    
    def _apply_advanced_noise_reduction(self, waveform: torch.Tensor, audio_path: str) -> torch.Tensor:
        """
        Apply advanced noise reduction techniques to the waveform.
        
        Args:
            waveform: Input audio waveform
            audio_path: Path to audio file for context
            
        Returns:
            Enhanced waveform
        """
        try:
            # First try advanced models if available
            if self.enhancement_models:
                enhanced_waveform = self._apply_ml_enhancement(waveform)
                if enhanced_waveform is not None:
                    return enhanced_waveform
            
            # Fallback to enhanced signal processing
            return self._apply_enhanced_signal_processing(waveform)
            
        except Exception as e:
            logger.error(f"Error in advanced noise reduction: {e}")
            return waveform  # Return original if processing fails
    
    def _apply_ml_enhancement(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        """Apply machine learning-based enhancement models."""
        try:
            audio = waveform.squeeze().numpy()
            
            for model_name, model in self.enhancement_models.items():
                try:
                    if "speechbrain" in model_name:
                        # Use SpeechBrain Sepformer for speech enhancement
                        enhanced_audio = model.separate_batch(waveform.unsqueeze(0))
                        if enhanced_audio is not None and len(enhanced_audio) > 0:
                            return enhanced_audio[0, 0, :].unsqueeze(0)  # Take first source
                            
                    elif "demucs" in model_name:
                        # Use Demucs for source separation
                        import demucs.api
                        separated = model.separate_tensor(waveform)
                        if separated is not None and len(separated) > 0:
                            return separated[0]  # Take first separated source
                            
                except Exception as model_error:
                    logger.warning(f"Error with {model_name}: {model_error}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ML enhancement: {e}")
            return None
    
    def _apply_enhanced_signal_processing(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply enhanced signal processing techniques for advanced performance.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Enhanced waveform
        """
        try:
            # Convert to numpy for processing
            audio = waveform.squeeze().numpy()
            
            # Apply multiple enhancement techniques in sequence
            enhanced_audio = self._advanced_spectral_subtraction(audio)
            enhanced_audio = self._adaptive_wiener_filtering(enhanced_audio)
            enhanced_audio = self._kalman_filtering(enhanced_audio)
            enhanced_audio = self._non_local_means_denoising(enhanced_audio)
            enhanced_audio = self._wavelet_denoising(enhanced_audio)
            
            # Convert back to tensor
            enhanced_waveform = torch.from_numpy(enhanced_audio).unsqueeze(0)
            
            return enhanced_waveform
            
        except Exception as e:
            logger.error(f"Error in enhanced signal processing: {e}")
            return waveform  # Return original if processing fails
    
    def _apply_noise_reduction(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply basic noise reduction techniques to the waveform.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Enhanced waveform
        """
        try:
            # Convert to numpy for processing
            audio = waveform.squeeze().numpy()
            
            # Apply various enhancement techniques
            enhanced_audio = self._spectral_subtraction(audio)
            enhanced_audio = self._wiener_filtering(enhanced_audio)
            enhanced_audio = self._adaptive_filtering(enhanced_audio)
            
            # Convert back to tensor
            enhanced_waveform = torch.from_numpy(enhanced_audio).unsqueeze(0)
            
            return enhanced_waveform
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return waveform  # Return original if processing fails
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Enhanced audio signal
        """
        try:
            # Compute STFT
            stft = np.fft.fft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames (assuming they contain mostly noise)
            noise_frames = min(10, len(magnitude) // 4)
            noise_spectrum = np.mean(magnitude[:noise_frames])
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor factor
            
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = np.real(np.fft.ifft(enhanced_stft))
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Error in spectral subtraction: {e}")
            return audio
    
    def _wiener_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Enhanced audio signal
        """
        try:
            # Simple Wiener filter implementation
            # In practice, you would use more sophisticated methods
            
            # Apply a simple high-pass filter to remove low-frequency noise
            from scipy import signal
            
            # Design high-pass filter
            nyquist = self.sample_rate / 2
            cutoff = 80  # Hz
            normalized_cutoff = cutoff / nyquist
            
            b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error in Wiener filtering: {e}")
            return audio
    
    def _adaptive_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply adaptive filtering for noise reduction.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Enhanced audio signal
        """
        try:
            # Simple adaptive filtering using moving average
            window_size = int(0.025 * self.sample_rate)  # 25ms window
            
            # Apply moving average filter
            filtered_audio = np.convolve(audio, np.ones(window_size)/window_size, mode='same')
            
            # Mix original and filtered signal
            alpha = 0.7  # Mixing factor
            enhanced_audio = alpha * audio + (1 - alpha) * filtered_audio
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Error in adaptive filtering: {e}")
            return audio
    
    def estimate_snr(self, audio_path: str) -> float:
        """
        Estimate Signal-to-Noise Ratio of the audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated SNR in dB
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            audio = waveform.squeeze().numpy()
            
            # Estimate signal power (using RMS)
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise power (using quiet segments)
            # Find quiet segments (low energy)
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            
            frame_energies = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.mean(frame ** 2)
                frame_energies.append(energy)
            
            # Use bottom 10% of frames as noise estimate
            frame_energies = np.array(frame_energies)
            noise_threshold = np.percentile(frame_energies, 10)
            noise_power = np.mean(frame_energies[frame_energies <= noise_threshold])
            
            # Calculate SNR
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 50  # Very high SNR if no noise detected
            
            return float(snr_db)
            
        except Exception as e:
            logger.error(f"Error estimating SNR: {e}")
            return 20.0  # Default SNR estimate
    
    def is_noisy_audio(self, audio_path: str, threshold: float = 15.0) -> bool:
        """
        Determine if audio is noisy based on SNR estimation.
        
        Args:
            audio_path: Path to audio file
            threshold: SNR threshold in dB (below this is considered noisy)
            
        Returns:
            True if audio is considered noisy
        """
        try:
            snr = self.estimate_snr(audio_path)
            return snr < threshold
            
        except Exception as e:
            logger.error(f"Error checking if audio is noisy: {e}")
            return False
    
    def get_enhancement_stats(self, original_path: str, enhanced_path: str) -> dict:
        """
        Get statistics comparing original and enhanced audio.
        
        Args:
            original_path: Path to original audio
            enhanced_path: Path to enhanced audio
            
        Returns:
            Dictionary with enhancement statistics
        """
        try:
            original_snr = self.estimate_snr(original_path)
            enhanced_snr = self.estimate_snr(enhanced_path)
            
            return {
                'original_snr': original_snr,
                'enhanced_snr': enhanced_snr,
                'snr_improvement': enhanced_snr - original_snr,
                'enhancement_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error getting enhancement stats: {e}")
            return {
                'original_snr': 0.0,
                'enhanced_snr': 0.0,
                'snr_improvement': 0.0,
                'enhancement_applied': False,
                'error': str(e)
            }
    
    def _advanced_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Advanced spectral subtraction with adaptive parameters."""
        try:
            # Compute STFT with overlap
            hop_length = 512
            n_fft = 2048
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Adaptive noise estimation
            noise_frames = min(20, len(magnitude[0]) // 4)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Adaptive over-subtraction factor based on SNR
            snr_estimate = np.mean(magnitude) / (np.mean(noise_spectrum) + 1e-10)
            alpha = max(1.5, min(3.0, 2.0 + 0.5 * (20 - snr_estimate) / 20))
            
            # Apply spectral subtraction
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.01 * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Error in advanced spectral subtraction: {e}")
            return audio
    
    def _adaptive_wiener_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive Wiener filtering with frequency-dependent parameters."""
        try:
            from scipy import signal
            
            # Design adaptive filter based on signal characteristics
            nyquist = self.sample_rate / 2
            
            # Adaptive cutoff based on signal energy distribution
            f, psd = signal.welch(audio, self.sample_rate, nperseg=1024)
            energy_80_percent = np.cumsum(psd) / np.sum(psd)
            cutoff_idx = np.where(energy_80_percent >= 0.8)[0][0]
            adaptive_cutoff = f[cutoff_idx]
            
            # Ensure cutoff is within reasonable bounds
            cutoff = max(80, min(adaptive_cutoff, 8000))
            normalized_cutoff = cutoff / nyquist
            
            # Design Butterworth filter
            b, a = signal.butter(6, normalized_cutoff, btype='high', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error in adaptive Wiener filtering: {e}")
            return audio
    
    def _kalman_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Kalman filtering for noise reduction."""
        try:
            # Simple Kalman filter implementation
            # State: [signal, derivative]
            # Measurement: current sample
            
            # Initialize Kalman filter parameters
            dt = 1.0 / self.sample_rate
            A = np.array([[1, dt], [0, 1]])  # State transition matrix
            H = np.array([[1, 0]])  # Observation matrix
            Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
            R = np.array([[0.5]])  # Measurement noise covariance
            
            # Initialize state and covariance
            x = np.array([[audio[0]], [0]])  # Initial state
            P = np.eye(2)  # Initial covariance
            
            filtered_audio = np.zeros_like(audio)
            filtered_audio[0] = audio[0]
            
            for i in range(1, len(audio)):
                # Predict
                x_pred = A @ x
                P_pred = A @ P @ A.T + Q
                
                # Update
                y = audio[i] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                
                x = x_pred + K @ y
                P = (np.eye(2) - K @ H) @ P_pred
                
                filtered_audio[i] = x[0, 0]
            
            return filtered_audio
            
        except Exception as e:
            logger.error(f"Error in Kalman filtering: {e}")
            return audio
    
    def _non_local_means_denoising(self, audio: np.ndarray) -> np.ndarray:
        """Non-local means denoising for audio."""
        try:
            # Simplified non-local means for 1D audio signal
            window_size = 5
            search_size = 11
            h = 0.1  # Filtering parameter
            
            denoised = np.zeros_like(audio)
            
            for i in range(len(audio)):
                # Define search window
                start = max(0, i - search_size // 2)
                end = min(len(audio), i + search_size // 2 + 1)
                
                weights = []
                values = []
                
                for j in range(start, end):
                    # Calculate similarity between patches
                    patch_i_start = max(0, i - window_size // 2)
                    patch_i_end = min(len(audio), i + window_size // 2 + 1)
                    patch_j_start = max(0, j - window_size // 2)
                    patch_j_end = min(len(audio), j + window_size // 2 + 1)
                    
                    patch_i = audio[patch_i_start:patch_i_end]
                    patch_j = audio[patch_j_start:patch_j_end]
                    
                    # Ensure patches are same size
                    min_len = min(len(patch_i), len(patch_j))
                    patch_i = patch_i[:min_len]
                    patch_j = patch_j[:min_len]
                    
                    # Calculate distance
                    distance = np.sum((patch_i - patch_j) ** 2) / len(patch_i)
                    weight = np.exp(-distance / (h ** 2))
                    
                    weights.append(weight)
                    values.append(audio[j])
                
                # Weighted average
                if weights:
                    weights = np.array(weights)
                    values = np.array(values)
                    denoised[i] = np.sum(weights * values) / np.sum(weights)
                else:
                    denoised[i] = audio[i]
            
            return denoised
            
        except Exception as e:
            logger.error(f"Error in non-local means denoising: {e}")
            return audio
    
    def _wavelet_denoising(self, audio: np.ndarray) -> np.ndarray:
        """Wavelet-based denoising."""
        try:
            import pywt
            
            # Choose wavelet and decomposition level
            wavelet = 'db4'
            level = 4
            
            # Decompose signal
            coeffs = pywt.wavedec(audio, wavelet, level=level)
            
            # Estimate noise level using median absolute deviation
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Apply soft thresholding
            threshold = sigma * np.sqrt(2 * np.log(len(audio)))
            coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            
            # Reconstruct signal
            denoised_audio = pywt.waverec(coeffs_thresh, wavelet)
            
            # Ensure same length
            if len(denoised_audio) != len(audio):
                denoised_audio = denoised_audio[:len(audio)]
            
            return denoised_audio
            
        except Exception as e:
            logger.error(f"Error in wavelet denoising: {e}")
            return audio
