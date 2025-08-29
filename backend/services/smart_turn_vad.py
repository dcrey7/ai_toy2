"""
Smart Turn v2 VAD Service for semantic voice activity detection
Uses Pipecat's Smart Turn v2 model to detect when user has finished speaking
"""

import torch
import numpy as np
import logging
from transformers import pipeline
import asyncio
from typing import Optional, List, Tuple
import time
import threading
from collections import deque

logger = logging.getLogger(__name__)

class SmartTurnVADService:
    def __init__(self, model_name="pipecat-ai/smart-turn-v2", device="cuda", buffer_duration=8.0, max_speech_duration=30.0):
        self.model_name = model_name
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.buffer_duration = buffer_duration  # Smart Turn v2 analyzes 8s windows
        self.sample_rate = 16000  # Smart Turn v2 expects 16kHz
        
        # Separate buffers for VAD analysis and STT transcription
        # VAD buffer: Rolling window for turn detection (8s max)
        self.vad_buffer = deque(maxlen=int(self.buffer_duration * self.sample_rate))
        # STT buffer: Complete speech from start to turn completion 
        # Max 60s to prevent memory issues (60s * 16kHz = 960k samples)
        self.max_stt_duration = 60.0  # Max 60 seconds of speech
        self.stt_buffer = deque()
        self.buffer_lock = threading.Lock()
        
        # VAD pipeline
        self.vad_pipeline: Optional[pipeline] = None
        self.last_turn_check = 0
        self.turn_check_interval = 0.2  # Check every 200ms for faster responsiveness  
        self.confidence_threshold = 0.35  # Optimized threshold for accuracy vs speed
        
        # State tracking
        self.is_processing = False
        self.consecutive_complete_detections = 0
        self.required_consecutive_detections = 1  # Only need 1 detection for faster response
        
        # Fallback mechanism
        self.speech_start_time = None
        self.max_speech_duration = max_speech_duration  # Configurable max speech duration
        self.silence_timeout = 3.0  # Complete after 3 seconds of silence (more forgiving)
        
        self._load_model()

    def _load_model(self):
        """Load Smart Turn v2 VAD model"""
        try:
            logger.info(f"Loading Smart Turn v2 VAD model: {self.model_name}")
            
            self.vad_pipeline = pipeline(
                "audio-classification",
                model=self.model_name,
                feature_extractor="facebook/wav2vec2-base",
                device=0 if self.device == "cuda" else -1,  # 0 for GPU, -1 for CPU
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("✅ Smart Turn v2 VAD model loaded successfully")
            logger.info(f"🎯 Using device: {self.device}, confidence threshold: {self.confidence_threshold}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Smart Turn v2 VAD model: {e}", exc_info=True)
            raise

    def add_pcm_chunk(self, pcm_data: np.ndarray) -> None:
        """
        Add PCM audio chunk to buffer (thread-safe)
        
        Args:
            pcm_data: numpy array of int16 or float32 PCM samples at 16kHz
        """
        # Convert to float32 in range [-1, 1] if needed
        if pcm_data.dtype == np.int16:
            audio_float = pcm_data.astype(np.float32) / 32768.0
        else:
            audio_float = pcm_data.astype(np.float32)
        
        # Check for speech activity and track timing
        has_speech = np.max(np.abs(audio_float)) > 0.01  # Simple activity detection
        current_time = time.time()
        
        if has_speech and self.speech_start_time is None:
            self.speech_start_time = current_time
            logger.info(f"🎤 Speech started at {current_time}")
        
        # Thread-safe buffer update - add to both buffers
        with self.buffer_lock:
            self.vad_buffer.extend(audio_float)  # Rolling window for VAD
            self.stt_buffer.extend(audio_float)  # Complete audio for STT
            
            # Prevent STT buffer from growing too large (max 60s)
            max_stt_samples = int(self.max_stt_duration * self.sample_rate)
            if len(self.stt_buffer) > max_stt_samples:
                # Keep the most recent audio for STT
                excess = len(self.stt_buffer) - max_stt_samples
                for _ in range(excess):
                    self.stt_buffer.popleft()
        
        logger.debug(f"Added {len(audio_float)} samples, VAD buffer: {len(self.vad_buffer)} ({len(self.vad_buffer) / self.sample_rate:.1f}s), STT buffer: {len(self.stt_buffer)} ({len(self.stt_buffer) / self.sample_rate:.1f}s), speech={has_speech}")

    def get_buffered_audio(self) -> np.ndarray:
        """Get complete STT audio buffer as numpy array (thread-safe)"""
        with self.buffer_lock:
            if not self.stt_buffer:
                return np.array([], dtype=np.float32)
            return np.array(list(self.stt_buffer), dtype=np.float32)
    
    def get_vad_audio(self) -> np.ndarray:
        """Get VAD analysis buffer (recent 8s window) as numpy array (thread-safe)"""
        with self.buffer_lock:
            if not self.vad_buffer:
                return np.array([], dtype=np.float32)
            return np.array(list(self.vad_buffer), dtype=np.float32)

    def is_turn_complete(self) -> Tuple[bool, float, str]:
        """
        Check if the user has completed their turn using Smart Turn v2
        
        Returns:
            tuple: (is_complete, confidence_score, reason)
        """
        if not self.vad_pipeline or self.is_processing:
            return False, 0.0, "VAD not ready or processing"
        
        # Rate limiting - don't check too frequently
        current_time = time.time()
        if current_time - self.last_turn_check < self.turn_check_interval:
            return False, 0.0, "Rate limited"
        
        self.last_turn_check = current_time
        
        # Get current VAD buffer (recent 8s window for analysis)
        audio_data = self.get_vad_audio()
        
        # Need at least 1.5 seconds of audio for meaningful analysis
        min_samples = int(1.5 * self.sample_rate)
        if len(audio_data) < min_samples:
            return False, 0.0, f"Insufficient audio: {len(audio_data)/self.sample_rate:.1f}s < 1.5s"
        
        # Fallback: Force completion after max speech duration with progressive warnings
        if self.speech_start_time:
            speech_duration = current_time - self.speech_start_time
            
            # Warn when approaching time limit
            warning_threshold = self.max_speech_duration * 0.8  # Warn at 80% of limit
            if speech_duration > warning_threshold and not hasattr(self, '_warned'):
                self._warned = True
                logger.info(f"⏰ Long speech detected: {speech_duration:.1f}s (approaching {self.max_speech_duration}s limit)")
            
            if speech_duration > self.max_speech_duration:
                logger.info(f"⏰ TIMEOUT: Forcing turn completion after {speech_duration:.1f}s")
                self.speech_start_time = None
                self._warned = False  # Reset warning flag
                return True, 0.8, "Timeout completion"
        
        try:
            self.is_processing = True
            
            # Use last 8 seconds for analysis (Smart Turn v2 optimal window)
            max_samples = int(8.0 * self.sample_rate)
            if len(audio_data) > max_samples:
                audio_data = audio_data[-max_samples:]
            
            # Run Smart Turn v2 VAD
            result = self.vad_pipeline(audio_data, top_k=None)[0]
            
            is_complete = result['label'] == 'complete'
            confidence = result['score']
            
            # Log all VAD results for debugging
            logger.info(f"🔍 VAD Analysis: label='{result['label']}', confidence={confidence:.3f}, threshold={self.confidence_threshold}, audio={len(audio_data)/self.sample_rate:.1f}s")
            
            # More permissive detection - accept either "complete" or high confidence on any label
            if (is_complete and confidence >= self.confidence_threshold) or confidence >= 0.7:
                self.consecutive_complete_detections += 1
                if self.consecutive_complete_detections >= self.required_consecutive_detections:
                    logger.info(f"🎯 Turn COMPLETE detected! (label: {result['label']}, confidence: {confidence:.3f})")
                    self.consecutive_complete_detections = 0  # Reset counter
                    # Don't clear buffer yet - STT needs it
                    return True, confidence, "Turn complete"
                else:
                    logger.info(f"🔄 Turn possibly complete (confidence: {confidence:.3f}, consecutive: {self.consecutive_complete_detections}/{self.required_consecutive_detections})")
                    return False, confidence, f"Needs more detections"
            else:
                # Reset consecutive counter if not complete
                self.consecutive_complete_detections = 0
                logger.info(f"🎤 Turn in progress: {result['label']} (confidence: {confidence:.3f})")
                return False, confidence, f"Turn {result['label']}"
            
        except Exception as e:
            logger.error(f"Smart Turn v2 VAD error: {e}")
            return False, 0.0, f"Error: {e}"
        finally:
            self.is_processing = False

    def clear_buffer(self):
        """Clear both audio buffers (thread-safe)"""
        with self.buffer_lock:
            self.vad_buffer.clear()
            self.stt_buffer.clear()
        self.consecutive_complete_detections = 0
        self.speech_start_time = None  # Reset speech timing
        self._warned = False  # Reset warning flag
        logger.debug("🧹 Audio buffers cleared")

    def get_buffer_duration(self) -> float:
        """Get current STT buffer duration in seconds"""
        with self.buffer_lock:
            return len(self.stt_buffer) / self.sample_rate

    def get_buffer_info(self) -> dict:
        """Get detailed buffer information for debugging"""
        with self.buffer_lock:
            vad_buffer_size = len(self.vad_buffer)
            stt_buffer_size = len(self.stt_buffer)
            vad_duration = vad_buffer_size / self.sample_rate
            stt_duration = stt_buffer_size / self.sample_rate
            return {
                "vad_buffer_size_samples": vad_buffer_size,
                "vad_buffer_duration_seconds": vad_duration,
                "stt_buffer_size_samples": stt_buffer_size,
                "stt_buffer_duration_seconds": stt_duration,
                "sample_rate": self.sample_rate,
                "consecutive_detections": self.consecutive_complete_detections,
                "confidence_threshold": self.confidence_threshold,
                "is_processing": self.is_processing
            }

    async def process_pcm_stream(self, pcm_chunk: np.ndarray) -> Tuple[bool, float, str]:
        """
        Process PCM audio chunk and check for turn completion (async wrapper)
        
        Args:
            pcm_chunk: PCM audio chunk as numpy array (int16 or float32)
            
        Returns:
            tuple: (is_complete, confidence, reason)
        """
        # Add chunk to buffer
        self.add_pcm_chunk(pcm_chunk)
        
        # Check if turn is complete (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.is_turn_complete)

    def reset_state(self):
        """Reset VAD state for new conversation"""
        self.clear_buffer()
        self.consecutive_complete_detections = 0
        self.last_turn_check = 0
        self._warned = False  # Reset warning flag
        logger.info("🔄 Smart Turn v2 VAD state reset")

    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up Smart Turn v2 VAD service...")
        
        if self.vad_pipeline:
            del self.vad_pipeline
            self.vad_pipeline = None
        
        self.clear_buffer()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("✅ Smart Turn v2 VAD service cleaned up")