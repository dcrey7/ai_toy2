
"""
Local Text-to-Speech Service with Kokoro TTS + espeak fallback
Optimized for RTX 3050 6GB VRAM
"""

import torch
import subprocess
import tempfile
import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalTTSService:
    def __init__(self, primary_engine="kokoro", device="cuda"):
        self.primary_engine = primary_engine
        self.device = device if torch.cuda.is_available() else "cpu"
        self.kokoro_pipeline = None
        self.sample_rate = 24000  # Kokoro's native sample rate
        self.espeak_available = False
        
        # Setup TTS engines
        self._setup_tts()
        
    def _setup_tts(self):
        """Setup TTS engines with fallback"""
        if self.primary_engine == "kokoro":
            try:
                self._load_kokoro()
                logger.info("✅ Kokoro TTS loaded successfully")
            except Exception as e:
                logger.warning(f"❌ Kokoro TTS failed to load: {e}")
                logger.info("Falling back to espeak TTS")
                self.primary_engine = "espeak"
        
        # Always verify espeak is available as fallback
        self._verify_espeak()
    
    def _load_kokoro(self):
        """Load Kokoro TTS model"""
        try:
            # Fix espeak-ng path issues by using system-installed espeak-ng
            import os
            
            # Set environment variables to use system espeak-ng
            os.environ["PHONEMIZER_ESPEAK_PATH"] = "/usr/bin/espeak-ng"
            os.environ["ESPEAK_DATA_PATH"] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
            
            # Try to configure espeakng-loader if available
            try:
                import espeakng_loader
                from phonemizer.backend.espeak.wrapper import EspeakWrapper
                
                # Override with system paths
                system_library = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"
                system_data = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
                
                # Check if system espeak-ng exists
                if os.path.exists(system_library) and os.path.exists(system_data):
                    EspeakWrapper.library_path = system_library
                    EspeakWrapper.data_path = system_data
                    logger.info(f"Using system espeak-ng: library={system_library}, data={system_data}")
                else:
                    # Fallback to espeakng-loader paths
                    library_path = espeakng_loader.get_library_path()
                    data_path = espeakng_loader.get_data_path()
                    espeakng_loader.make_library_available()
                    EspeakWrapper.library_path = library_path
                    EspeakWrapper.data_path = data_path
                    logger.info(f"Using espeakng-loader: library={library_path}, data={data_path}")
                
                # Add set_data_path method if missing
                if not hasattr(EspeakWrapper, 'set_data_path'):
                    def set_data_path_dummy(cls, path):
                        cls.data_path = path
                    EspeakWrapper.set_data_path = classmethod(set_data_path_dummy)
                    
            except Exception as espeak_err:
                logger.warning(f"EspeakWrapper configuration warning: {espeak_err}")
            
            # Now try to import and initialize Kokoro
            from kokoro import KPipeline
            
            # Initialize with American English voice code 'a'
            self.kokoro_pipeline = KPipeline(lang_code='a')
            logger.info("✅ Kokoro TTS pipeline initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Kokoro package not installed: {e}, falling back to espeak")
            self.primary_engine = "espeak"
            self.kokoro_pipeline = None
        except Exception as e:
            logger.error(f"❌ Kokoro initialization failed: {e}")
            logger.info("🔄 Falling back to espeak TTS engine")
            self.primary_engine = "espeak"
            self.kokoro_pipeline = None
    
    def _verify_espeak(self):
        """Verify espeak is available"""
        try:
            subprocess.run(["espeak", "--version"], capture_output=True, check=True)
            self.espeak_available = True
            logger.info("✅ espeak TTS available as fallback")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ espeak not available! Please install: sudo apt install espeak")
            self.espeak_available = False
    
    def synthesize(self, text, voice="en_heart", force_fallback=False):
        """
        Convert text to speech audio
        
        Args:
            text (str): Text to synthesize
            voice (str): Voice to use (default: 'en_heart' for Kokoro)
            force_fallback (bool): If true, forces use of espeak
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        if not text.strip():
            logger.warning("Empty text provided for TTS")
            return np.array([]), self.sample_rate
            
        logger.debug(f"TTS request: text='{text}', voice='{voice}', engine={self.primary_engine}, force_fallback={force_fallback}")
        
        # Try primary engine first
        if not force_fallback and self.primary_engine == "kokoro" and self.kokoro_pipeline is not None:
            try:
                return self._synthesize_kokoro(text, voice)
            except Exception as e:
                logger.warning(f"Kokoro synthesis failed: {e}, falling back to espeak")
        
        # Fallback to espeak
        if self.espeak_available:
            return self._synthesize_espeak(text)
        else:
            logger.error("No TTS engine available")
            return np.zeros(8000, dtype=np.float32), 16000
    
    def _synthesize_kokoro(self, text, voice="en_heart"):
        """Synthesize using Kokoro TTS"""
        try:
            # Generate audio using Kokoro
            generator = self.kokoro_pipeline(text, voice=voice)
            
            # Get the first (and likely only) result
            for i, (gs, ps, audio) in enumerate(generator):
                if i == 0:  # Take first result
                    # Convert to numpy array if needed
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    # Ensure float32 format
                    audio = audio.astype(np.float32)
                    
                    # Normalize audio if needed (Kokoro outputs normalized [-1, 1])
                    if np.max(np.abs(audio)) > 1.0:
                        audio = audio / np.max(np.abs(audio))
                    
                    logger.debug(f"Kokoro generated {len(audio)} samples at {self.sample_rate}Hz")
                    return audio, self.sample_rate
                    
            logger.warning("Kokoro generator returned no audio")
            raise RuntimeError("No audio generated by Kokoro")
            
        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            raise
    
    def _synthesize_espeak(self, text):
        """Synthesize using espeak (fallback)"""
        try:
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Run espeak to generate audio file
            cmd = [
                "espeak",
                "-s", "150",  # Speed
                "-p", "50",   # Pitch
                "-w", tmp_path,  # Output to wav file
                text
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stderr:
                logger.warning(f"espeak stderr: {result.stderr}")
            
            # Read the generated audio file
            import soundfile as sf
            audio_data, sample_rate = sf.read(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Convert to 24000Hz if needed to match Kokoro
            if sample_rate != self.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                sample_rate = self.sample_rate
            
            logger.debug(f"espeak generated {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data.astype(np.float32), sample_rate
            
        except Exception as e:
            logger.error(f"espeak synthesis error: {e}")
            # Return silence if everything fails
            return np.zeros(8000, dtype=np.float32), self.sample_rate
    
    def cleanup(self):
        """Clean up resources"""
        if self.kokoro_pipeline is not None:
            del self.kokoro_pipeline
            self.kokoro_pipeline = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("TTS model cleaned up")
