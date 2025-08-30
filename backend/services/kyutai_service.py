"""
Kyutai STT Service for streaming transcription.
"""

import torch
from transformers import (
    KyutaiSpeechToTextProcessor,
    KyutaiSpeechToTextForConditionalGeneration
)
import numpy as np
import logging
import librosa

logger = logging.getLogger(__name__)

class KyutaiSTTService:
    def __init__(self, model_name="kyutai/stt-1b-en_fr-trfs", device="cuda"):
        self.model_name = model_name
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self.processor = None
        self.model = None
        self.target_sample_rate = 24000  # Kyutai models expect 24kHz
        self._load_model()

    def _load_model(self):
        """Load the Kyutai STT model using proper classes from Hugging Face."""
        try:
            logger.info(f"Loading Kyutai STT model {self.model_name} on device {self.device}")
            
            # Clear CUDA cache before loading
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache before model loading")
            
            # Load processor and model separately using the dedicated Kyutai classes
            self.processor = KyutaiSpeechToTextProcessor.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Try to move to device with fallback to CPU
            try:
                self.model = self.model.to(self.torch_device)
                logger.info(f"✅ Model successfully moved to {self.device}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as cuda_err:
                logger.warning(f"⚠️ CUDA device busy/unavailable: {cuda_err}")
                logger.info("🔄 Falling back to CPU device")
                self.device = "cpu"
                self.torch_device = torch.device("cpu")
                self.model = self.model.to(self.torch_device)
                logger.info("✅ Model successfully moved to CPU")
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("✅ Kyutai STT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Kyutai STT model: {e}", exc_info=True)
            raise

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio buffer.
        
        Args:
            audio_data: numpy array of audio samples (float32)
            sample_rate: sample rate of audio
            
        Returns:
            str: transcribed text
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Kyutai STT model not loaded")
            
        try:
            # Convert sample rate to 24kHz if needed (Kyutai expects 24kHz)
            if sample_rate != self.target_sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sample_rate
                )
                sample_rate = self.target_sample_rate
            
            # Process the audio using the Kyutai processor
            inputs = self.processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the generated tokens
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            logger.debug(f"Kyutai STT result: '{transcription}'")
            return transcription
            
        except Exception as e:
            logger.error(f"Kyutai STT transcription failed: {e}", exc_info=True)
            return ""

    def cleanup(self):
        """Clean up resources to free VRAM."""
        logger.info("Cleaning up Kyutai STT model...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Kyutai STT model cleaned up.")
