
"""
Local Speech-to-Text Service using Whisper Small
Optimized for RTX 3050 6GB VRAM with Silero VAD integration
"""

import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

class LocalSTTService:
    def __init__(self, model_name="openai/whisper-small", device="cuda", enable_vad=True):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_vad = enable_vad
        self.pipeline = None
        self.vad_model = None
        self.transcription_buffer = ""
        self.last_transcription_time = 0
        self.vad_threshold = 0.3  # Lowered for better sensitivity
        self._load_model()
        if self.enable_vad:
            self._load_vad()
        
    def _load_model(self):
        """Load Whisper model with VRAM optimization"""
        try:
            logger.info(f"Loading STT model {self.model_name} on {self.device}")
            
            # Use pipeline for simplicity and optimization
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                torch_dtype=torch.float16,  # Half precision
                device=self.device,
                chunk_length_s=30,  # Optimal for small model
                model_kwargs={"attn_implementation": "sdpa"} if self.device == "cuda" else {}  # More compatible than flash_attention_2
            )
            
            logger.info("STT model loaded successfully")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise
    
    def _load_vad(self):
        """Load Silero VAD model"""
        try:
            logger.info("Loading Silero VAD model")
            import torch
            
            # Load Silero VAD model
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Extract utility functions
            self.get_speech_timestamps = utils[0]
            self.save_audio = utils[1]
            self.read_audio = utils[2]
            self.VADIterator = utils[3]
            self.collect_chunks = utils[4]
            
            logger.info("VAD model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load VAD model: {e}")
            self.enable_vad = False
            self.vad_model = None
    
    def has_speech(self, audio_data, sample_rate=16000):
        """Check if audio contains speech using VAD"""
        if not self.enable_vad or self.vad_model is None:
            return True  # Assume speech if no VAD
            
        try:
            # Ensure audio is torch tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            else:
                audio_tensor = audio_data.float()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=sample_rate,
                threshold=self.vad_threshold  # Use lower threshold
            )
            
            logger.debug(f"VAD detected {len(speech_timestamps)} speech segments")
            return len(speech_timestamps) > 0
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return True  # Default to processing if VAD fails
    
    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio to text
        
        Args:
            audio_data: numpy array of audio samples
            sample_rate: sample rate of audio
            
        Returns:
            str: transcribed text
        """
        try:
            if self.pipeline is None:
                raise RuntimeError("STT model not loaded")
            
            # Check for speech first with VAD
            if not self.has_speech(audio_data, sample_rate):
                logger.debug("No speech detected by VAD")
                return ""  # No speech detected
                
            # Ensure audio is float32 numpy array
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Transcribe
            result = self.pipeline(
                {"sampling_rate": sample_rate, "raw": audio_data},
                return_timestamps=False,
                generate_kwargs={"max_new_tokens": 100}  # Limit for responsiveness
            )
            
            text = result["text"].strip()
            
            # Update transcription buffer for streaming
            current_time = time.time()
            if text and text != self.transcription_buffer:
                self.transcription_buffer = text
                self.last_transcription_time = current_time
            
            logger.debug(f"STT result: {text}")
            return text
            
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return ""
    
    async def transcribe_streaming(self, audio_data, sample_rate=16000, callback=None):
        """Streaming transcription with real-time updates"""
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self.transcribe, audio_data, sample_rate)
            
            # Call callback with result if provided
            if callback and text:
                await callback(text, is_final=True)
            
            logger.debug(f"Streaming transcription result: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return ""
    
    def get_transcription_buffer(self):
        """Get current transcription buffer for live display"""
        return self.transcription_buffer
    
    def _log_memory_usage(self):
        """Log VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM usage - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    def cleanup(self):
        """Clean up resources"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if self.vad_model is not None:
            del self.vad_model
            self.vad_model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("STT and VAD models cleaned up")
