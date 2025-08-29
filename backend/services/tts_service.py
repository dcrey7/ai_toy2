
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
        
        # Text chunking settings for long responses (optimized for RTX 3050 6GB)
        self.chunk_size = 180  # Smaller chunks for better VRAM management
        self.chunk_delimiters = ['. ', '! ', '? ', '; ', ': ', '\n']
        
        # Performance optimizations
        self.enable_memory_cleanup = True
        
        # Setup TTS engines
        self._setup_tts()
    
    def _chunk_text(self, text):
        """Split text into manageable chunks for TTS processing"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        remaining = text
        
        while remaining:
            if len(remaining) <= self.chunk_size:
                chunks.append(remaining.strip())
                break
            
            # Find the best split point within chunk_size
            best_split = -1
            for delimiter in self.chunk_delimiters:
                # Look for delimiter within chunk_size characters
                split_pos = remaining.rfind(delimiter, 0, self.chunk_size)
                if split_pos > best_split:
                    best_split = split_pos + len(delimiter)
            
            if best_split > 0:
                # Split at the delimiter
                chunks.append(remaining[:best_split].strip())
                remaining = remaining[best_split:].strip()
            else:
                # No good split point found, force split at chunk_size
                chunks.append(remaining[:self.chunk_size].strip())
                remaining = remaining[self.chunk_size:].strip()
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        logger.debug(f"Split text into {len(chunks)} chunks: {[len(c) for c in chunks]} chars each")
        return chunks
        
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
        Convert text to speech audio with automatic chunking for long text
        
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
            
        logger.debug(f"TTS request: text='{text[:50]}...', voice='{voice}', engine={self.primary_engine}, force_fallback={force_fallback}")
        
        # Split text into chunks for better processing
        chunks = self._chunk_text(text)
        logger.info(f"🔄 Processing {len(chunks)} text chunks for TTS")
        
        all_audio = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
            
            # Try primary engine first
            if not force_fallback and self.primary_engine == "kokoro" and self.kokoro_pipeline is not None:
                try:
                    audio, sample_rate = self._synthesize_kokoro(chunk, voice)
                    if len(audio) > 0:
                        all_audio.append(audio)
                        
                        # Clear CUDA cache after each chunk to prevent VRAM buildup
                        if self.enable_memory_cleanup and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                except Exception as e:
                    logger.warning(f"Kokoro synthesis failed for chunk {i+1}: {e}, falling back to espeak")
            
            # Fallback to espeak for this chunk
            if self.espeak_available:
                audio, sample_rate = self._synthesize_espeak(chunk)
                if len(audio) > 0:
                    all_audio.append(audio)
            else:
                logger.error(f"No TTS engine available for chunk {i+1}")
        
        if not all_audio:
            logger.error("No audio generated from any chunks")
            return np.zeros(8000, dtype=np.float32), self.sample_rate
        
        # Concatenate all audio chunks with small gaps for natural flow
        gap_samples = int(0.2 * self.sample_rate)  # 200ms gap between chunks for faster speech
        gap_audio = np.zeros(gap_samples, dtype=np.float32)
        
        concatenated_audio = []
        for i, audio in enumerate(all_audio):
            concatenated_audio.append(audio)
            if i < len(all_audio) - 1:  # Don't add gap after last chunk
                concatenated_audio.append(gap_audio)
        
        final_audio = np.concatenate(concatenated_audio)
        logger.info(f"✅ Generated {len(final_audio)} samples ({len(final_audio)/self.sample_rate:.1f}s) from {len(chunks)} chunks")
        
        return final_audio, self.sample_rate
    
    def _synthesize_kokoro(self, text, voice="en_heart"):
        """Synthesize using Kokoro TTS - processes all generator results"""
        try:
            # Generate audio using Kokoro
            generator = self.kokoro_pipeline(text, voice=voice)
            
            # Collect ALL audio segments from generator
            audio_segments = []
            for i, (gs, ps, audio) in enumerate(generator):
                # Convert to numpy array if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Ensure float32 format
                audio = audio.astype(np.float32)
                
                # Normalize audio if needed (Kokoro outputs normalized [-1, 1])
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / np.max(np.abs(audio))
                
                audio_segments.append(audio)
                logger.debug(f"Kokoro segment {i+1}: {len(audio)} samples at {self.sample_rate}Hz")
            
            if not audio_segments:
                logger.warning("Kokoro generator returned no audio")
                raise RuntimeError("No audio generated by Kokoro")
            
            # Concatenate all segments for this chunk
            full_audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
            logger.debug(f"Kokoro total: {len(full_audio)} samples ({len(full_audio)/self.sample_rate:.1f}s) from {len(audio_segments)} segments")
            return full_audio, self.sample_rate
            
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
