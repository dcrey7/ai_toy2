#!/usr/bin/env python3
"""
AI Voice Assistant - v2.2 (Streaming VAD & UI Logging)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import asyncio
import json
import logging
import ssl
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import yaml
import base64
import numpy as np
import datetime
import re
import io
from pydub import AudioSegment

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from starlette.websockets import WebSocketState
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(str(Path(__file__).parent / "backend"))
from services.kyutai_service import KyutaiSTTService
from services.tts_service import LocalTTSService
from services.ollama_service import OllamaService
from services.smart_turn_vad import SmartTurnVADService

# Setup root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Agent & System Architecture ---

class Agent:
    def __init__(self, app: 'VoiceAssistant'):
        self.app = app
        self._task: Optional[asyncio.Task] = None

    async def run(self):
        raise NotImplementedError

    def start(self):
        logger.info(f"Starting {self.__class__.__name__}...")
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        logger.info(f"{self.__class__.__name__} stopped.")

class WebSocketAgent(Agent):
    """Handles WebSocket communication with Smart Turn v2 hands-free recording."""
    def __init__(self, app: 'VoiceAssistant', websocket: WebSocket):
        super().__init__(app)
        self.websocket = websocket
        self.client_id = str(id(websocket))
        self.voice_mode_enabled = False
        self.is_streaming_audio = False
        
        # Smart Turn v2 VAD service for this client
        self.vad_service: Optional[SmartTurnVADService] = None
        self.turn_detection_task: Optional[asyncio.Task] = None

    async def _start_vad_service(self):
        """Initialize Smart Turn v2 VAD service for this client"""
        try:
            if self.vad_service is None:
                self.vad_service = SmartTurnVADService(
                    model_name="pipecat-ai/smart-turn-v2",
                    device="cuda"
                )
                await self.app.comms_out_queue.put({
                    "type": "log", 
                    "level": "success", 
                    "message": "✅ Smart Turn v2 VAD service initialized",
                    "websocket": self.websocket
                })
        except Exception as e:
            await self.app.comms_out_queue.put({
                "type": "log", 
                "level": "error", 
                "message": f"❌ VAD service init failed: {e}",
                "websocket": self.websocket
            })
            raise

    async def _continuous_turn_detection(self):
        """Continuous turn detection loop for Smart Turn v2"""
        try:
            # Wait briefly for TTS to start, then begin VAD processing
            await asyncio.sleep(0.5)  # Short delay to let TTS begin
            
            while self.voice_mode_enabled and self.is_streaming_audio:
                if self.vad_service:
                    is_complete, confidence, reason = self.vad_service.is_turn_complete()
                    
                    if is_complete:
                        # Temporarily disable audio streaming to prevent feedback
                        self.is_streaming_audio = False
                        
                        await self.app.comms_out_queue.put({
                            "type": "log",
                            "level": "success",
                            "message": f"🎯 Turn complete detected (confidence: {confidence:.3f})",
                            "websocket": self.websocket
                        })
                        
                        # Update VAD status to processing
                        await self.app.comms_out_queue.put({
                            "type": "vad_status", 
                            "status": "processing", 
                            "websocket": self.websocket
                        })
                        
                        # Get accumulated audio buffer and send to STT
                        audio_buffer = self.vad_service.get_buffered_audio()
                        if len(audio_buffer) > 0:
                            await self.app.stt_in_queue.put({
                                "audio_data": audio_buffer,
                                "websocket": self.websocket,
                                "source": "smart_turn_vad",
                                "sample_rate": 16000
                            })
                            
                            # Clear buffer after sending to STT
                            self.vad_service.clear_buffer()
                        else:
                            logger.warning("⚠️ No audio buffer available for STT")
                        
                        # Wait before re-enabling to prevent immediate feedback
                        await asyncio.sleep(2.0)
                        
                        # Re-enable streaming after AI response
                        if self.voice_mode_enabled:
                            self.is_streaming_audio = True
                            await self.app.comms_out_queue.put({
                                "type": "vad_status", 
                                "status": "inactive", 
                                "websocket": self.websocket
                            })
                
                # Check every 100ms for responsiveness
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.app.comms_out_queue.put({
                "type": "log",
                "level": "error",
                "message": f"❌ Turn detection error: {e}",
                "websocket": self.websocket
            })

    async def run(self):
        try:
            # Send initial status on connect
            await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "🟢 Ready", "websocket": self.websocket})
            await self.app.comms_out_queue.put({"type": "status", "component": "tts", "status": "🟢 Ready", "websocket": self.websocket})
            await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "🟢 Ready", "websocket": self.websocket})
            try:
                import torch
                if torch.cuda.is_available():
                    await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": f"🟢 {torch.cuda.get_device_name(0)}", "websocket": self.websocket})
                else:
                    await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": "⚪ CPU Only", "websocket": self.websocket})
            except ImportError:
                await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": "⚪ N/A", "websocket": self.websocket})

            while True:
                message = json.loads(await self.websocket.receive_text())
                msg_type = message.get("type")

                if msg_type == 'toggle_voice_mode':
                    self.voice_mode_enabled = message.get('enabled', False)
                    if self.voice_mode_enabled:
                        # Initialize VAD service and start continuous turn detection
                        await self._start_vad_service()
                        self.is_streaming_audio = True
                        self.turn_detection_task = asyncio.create_task(self._continuous_turn_detection())
                        
                        # Send greeting
                        greeting_text = "Hi! I'm your AI voice assistant. How can I help you?"
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 Voice mode enabled - hands-free conversation ready", "websocket": self.websocket})
                        await self.app.comms_out_queue.put({"type": "response", "text": greeting_text, "websocket": self.websocket})
                        await self.app.tts_in_queue.put({"text": greeting_text, "websocket": self.websocket, "use_fallback": False, "is_greeting": True})
                        
                    else:
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🔇 Voice mode disabled", "websocket": self.websocket})
                        
                        # Clean up VAD service and tasks
                        self.is_streaming_audio = False
                        if self.turn_detection_task and not self.turn_detection_task.done():
                            self.turn_detection_task.cancel()
                        if self.vad_service:
                            self.vad_service.cleanup()
                            self.vad_service = None

                elif msg_type == 'pcm_chunk' and self.voice_mode_enabled and self.is_streaming_audio:
                    # Real-time PCM audio chunk from frontend
                    try:
                        pcm_b64 = message.get('data', '')
                        sample_rate = message.get('sampleRate', 16000)
                        channels = message.get('channels', 1)
                        samples = message.get('samples', 0)
                        
                        if pcm_b64 and self.vad_service:
                            # Decode base64 PCM data
                            pcm_bytes = base64.b64decode(pcm_b64)
                            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                            
                            # Add to VAD service buffer
                            self.vad_service.add_pcm_chunk(pcm_array)
                            
                            # Update VAD status to show activity
                            await self.app.comms_out_queue.put({"type": "vad_status", "status": "speaking", "websocket": self.websocket})
                            
                    except Exception as e:
                        await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ PCM processing error: {e}", "websocket": self.websocket})
                

                elif msg_type == 'text_message':
                    user_text = message.get("text")
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"💬 Text message received: '{user_text}'", "websocket": self.websocket})
                    await self.app.llm_in_queue.put({"text": user_text, "websocket": self.websocket, "source": "text"})

        except WebSocketDisconnect:
            logger.info(f"Client {self.client_id} disconnected.")
        finally:
            # Clean up VAD service and tasks
            self.is_streaming_audio = False
            if self.turn_detection_task and not self.turn_detection_task.done():
                self.turn_detection_task.cancel()
            if self.vad_service:
                self.vad_service.cleanup()
                self.vad_service = None
            self.app.active_clients.pop(self.websocket, None)

class STTAgent(Agent):
    async def run(self):
        while True:
            job = await self.app.stt_in_queue.get()
            ws = job["websocket"]
            audio_data = job.get("audio_data")
            source = job.get("source", "unknown")
            sample_rate = job.get("sample_rate", 16000)
            
            try:
                await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "🟡 Transcribing...", "websocket": ws})
                
                if audio_data is None or (isinstance(audio_data, list) and len(audio_data) == 0):
                    raise Exception("No audio data received")
                
                # Handle different audio formats
                if source == "smart_turn_vad":
                    # Direct PCM data from Smart Turn v2 VAD (numpy array)
                    if isinstance(audio_data, np.ndarray):
                        samples = audio_data.astype(np.float32)
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"📝 Processing Smart Turn v2 VAD audio: {len(samples)} samples ({len(samples)/sample_rate:.1f}s)", "websocket": ws})
                    else:
                        raise Exception("Expected numpy array for Smart Turn v2 VAD data")
                        
                else:
                    # Legacy WebM format (list of byte chunks)
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"📝 Processing legacy WebM recording ({len(audio_data)} chunks)", "websocket": ws})
                    
                    import tempfile
                    combined_data = b''.join(audio_data)
                    temp_file = None
                    
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                        tmp.write(combined_data)
                        temp_file = tmp.name
                    
                    try:
                        audio_segment = AudioSegment.from_file(temp_file, format="webm")
                        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                        
                        await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ Successfully processed WebM: {len(samples)} samples ({len(samples)/16000:.1f}s)", "websocket": ws})
                        
                    except Exception as decode_error:
                        await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ Audio decoding failed: {decode_error}", "websocket": ws})
                        raise Exception(f"Audio decoding failed: {decode_error}")
                    finally:
                        # Clean up temp file
                        if temp_file and os.path.exists(temp_file):
                            os.unlink(temp_file)
                
                # Check if we have meaningful audio (more than 0.5 seconds)
                if len(samples) < sample_rate * 0.5:  # Less than 0.5 seconds
                    await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": f"⚠️ Audio too short for transcription: {len(samples)/sample_rate:.1f}s", "websocket": ws})
                    text = ""
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🔊 Transcribing {len(samples)/sample_rate:.1f}s of audio with Kyutai STT", "websocket": ws})
                    text = self.app.stt_service.transcribe(samples, sample_rate=sample_rate)
                
                if text and text.strip():
                    await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ STT transcription: '{text}'", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "transcript", "text": text, "is_final": True, "websocket": ws})
                    await self.app.llm_in_queue.put({"text": text, "websocket": ws, "source": "voice"})
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": "⚠️ STT returned empty transcription - Smart Turn v2 continues listening", "websocket": ws})
                        
            except Exception as e:
                logger.error(f"STTAgent error: {e}")
                await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ STT Error: {e}", "websocket": ws})
                # Smart Turn v2 continues listening automatically on error
            finally:
                await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "🟢 Ready", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "vad_status", "status": "idle", "websocket": ws})

class LLMAgent(Agent):
    async def run(self):
        while True:
            job = await self.app.llm_in_queue.get()
            ws, text, source = job["websocket"], job["text"], job.get("source", "unknown")
            try:
                await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "🟡 Thinking...", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🤖 LLM Agent processing: '{text}' (source: {source})", "websocket": ws})
                
                response_text = await self.app.llm_service.generate_response(text, context=self.app.conversation_history)
                
                self.app.conversation_history.append({"role": "user", "content": text})
                self.app.conversation_history.append({"role": "assistant", "content": response_text})
                self.app.conversation_history = self.app.conversation_history[-20:]
                
                await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ LLM generated response: '{response_text[:100]}...'", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "response", "text": response_text, "websocket": ws})
                
                # Only synthesize speech if the original input was voice
                if source == "voice":
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎵 Sending LLM response to TTS for voice synthesis", "websocket": ws})
                    await self.app.tts_in_queue.put({"text": response_text, "websocket": ws})
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "📝 Text mode - skipping TTS synthesis", "websocket": ws})
                    
            except Exception as e:
                logger.error(f"LLMAgent error: {e}")
                await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ LLM Error: {e}", "websocket": ws})
            finally:
                await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "🟢 Ready", "websocket": ws})

class TTSAgent(Agent):
    async def run(self):
        while True:
            job = await self.app.tts_in_queue.get()
            ws, text = job["websocket"], job["text"]
            use_fallback = job.get("use_fallback", False)
            is_greeting = job.get("is_greeting", False)
            try:
                await self.app.comms_out_queue.put({"type": "status", "component": "tts", "status": "🟡 Speaking...", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🎵 TTS Agent processing: '{text[:50]}...' (fallback={use_fallback})", "websocket": ws})
                
                # Sanitize text for TTS to avoid phonemizer errors with markdown
                sanitized_text = re.sub(r'[*`#_]', '', text)

                engine_name = "espeak" if use_fallback else self.app.tts_service.primary_engine
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🔊 Using TTS engine: {engine_name}", "websocket": ws})
                
                audio_data, sample_rate = self.app.tts_service.synthesize(sanitized_text, voice="af_heart", force_fallback=use_fallback)
                
                if audio_data is not None and len(audio_data) > 0:
                    await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ TTS generated {len(audio_data)} audio samples at {sample_rate}Hz", "websocket": ws})
                    audio_b64 = base64.b64encode((audio_data * 32767).astype(np.int16).tobytes()).decode('utf-8')
                    await self.app.comms_out_queue.put({"type": "tts_audio", "audio": audio_b64, "sample_rate": sample_rate, "text": text, "websocket": ws})
                    
                    # TTS completed successfully 
                    if is_greeting:
                        # After greeting TTS finishes, start VAD listening with a brief delay
                        await asyncio.sleep(1.0)  # Wait 1 second after greeting ends
                        await self.app.comms_out_queue.put({"type": "vad_status", "status": "inactive", "websocket": ws})
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 Greeting finished - Smart Turn v2 VAD now listening for your voice", "websocket": ws})
                    # Smart Turn v2 handles turn detection automatically for all other responses
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": "⚠️ TTS returned no audio data", "websocket": ws})
                    
            except Exception as e:
                logger.error(f"TTSAgent error: {e}")
                await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ TTS Error: {e}", "websocket": ws})
            finally:
                await self.app.comms_out_queue.put({"type": "status", "component": "tts", "status": "🟢 Ready", "websocket": ws})

class CommsAgent(Agent):
    async def run(self):
        while True:
            message = await self.app.comms_out_queue.get()
            if message.get("type") == "disconnect": continue
            websocket = message.pop("websocket")
            await self.app.send_to_client(websocket, message)

class QueueLogHandler(logging.Handler):
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
    def emit(self, record: logging.LogRecord):
        self.queue.put_nowait(self.format(record))

class LogAgent(Agent):
    def __init__(self, app: 'VoiceAssistant', queue: asyncio.Queue):
        super().__init__(app)
        self.queue = queue

    async def run(self):
        while True:
            record = await self.queue.get()
            message = {"type": "log", "level": "info", "message": record}
            for client_ws in self.app.active_clients.keys():
                await self.app.comms_out_queue.put({**message, "websocket": client_ws})

class VoiceAssistant:
    def __init__(self, log_queue: asyncio.Queue):
        self.app = FastAPI(title="AI Voice Assistant", version="2.2.0")
        self.log_queue = log_queue
        self.config: dict = {}
        self.active_clients: Dict[WebSocket, WebSocketAgent] = {}
        self.conversation_history: List[dict] = []
        self.stt_in_queue = asyncio.Queue()
        self.llm_in_queue = asyncio.Queue()
        self.tts_in_queue = asyncio.Queue()
        self.comms_out_queue = asyncio.Queue()
        self.stt_service: Optional[KyutaiSTTService] = None
        self.llm_service: Optional[OllamaService] = None
        self.tts_service: Optional[LocalTTSService] = None
        self.agents: List[Agent] = []

    async def load_config(self):
        config_file = Path(__file__).parent / "config" / "settings.yaml"
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}", exc_info=True)
            sys.exit(1)

    async def initialize_services(self):
        stt_config = self.config['models']['stt']
        self.stt_service = KyutaiSTTService(
            model_name=stt_config['model_name'],
            device=stt_config['device']
        )
        llm_config = self.config['models']['llm']
        self.llm_service = OllamaService(base_url=llm_config['base_url'], model_name=llm_config['model_name'])
        tts_config = self.config['models']['tts']
        self.tts_service = LocalTTSService(primary_engine=tts_config['primary'], device=tts_config.get('device', 'cuda'))

    def initialize_agents(self):
        self.agents = [STTAgent(self), LLMAgent(self), TTSAgent(self), CommsAgent(self), LogAgent(self, self.log_queue)]
        for agent in self.agents: agent.start()

    def setup_fastapi(self):
        self.app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        static_path = Path(__file__).parent / "static"
        self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        @self.app.get("/")
        async def root(): return FileResponse(static_path / "index.html")
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            ws_agent = WebSocketAgent(self, websocket)
            self.active_clients[websocket] = ws_agent
            await ws_agent.run()

    async def send_to_client(self, websocket: WebSocket, message: dict):
        try:
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send to client {id(websocket)}: {e}")

    async def cleanup(self):
        for agent in self.agents: await agent.stop()
        if self.stt_service: self.stt_service.cleanup()
        if self.tts_service: self.tts_service.cleanup()
        if self.llm_service: await self.llm_service.cleanup()

async def main():
    log_queue = asyncio.Queue()
    handler = QueueLogHandler(log_queue)
    logging.getLogger().addHandler(handler)

    assistant = VoiceAssistant(log_queue)
    
    await assistant.load_config()
    await assistant.initialize_services()
    assistant.initialize_agents()
    assistant.setup_fastapi()
    
    cert_file, key_file = create_ssl_certificate()
    
    # Configure server with or without SSL
    if cert_file and key_file:
        logger.info("🔒 Starting HTTPS server on https://localhost:8080")
        config = uvicorn.Config(assistant.app, host="0.0.0.0", port=8080, ssl_keyfile=key_file, ssl_certfile=cert_file, log_level="info")
    else:
        logger.info("⚠️  Starting HTTP server on http://localhost:8080 (not secure)")
        config = uvicorn.Config(assistant.app, host="0.0.0.0", port=8080, log_level="info")
    
    server = uvicorn.Server(config)
    
    try: await server.serve()
    finally: await assistant.cleanup()

def create_ssl_certificate():
    """Create self-signed SSL certificate for HTTPS"""
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime
    import ipaddress
    
    cert_dir = Path(__file__).parent / "certs"
    cert_dir.mkdir(exist_ok=True)
    cert_file = cert_dir / "localhost.pem"
    key_file = cert_dir / "localhost-key.pem"
    
    # Return existing certificates if they exist
    if cert_file.exists() and key_file.exists():
        logger.info("✅ Using existing SSL certificates")
        return str(cert_file), str(key_file)
    
    try:
        logger.info("🔐 Generating new SSL certificate...")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Voice Assistant"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.now(datetime.timezone.utc)
        ).not_valid_after(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write private key
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Write certificate
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        logger.info("✅ SSL certificate generated successfully")
        logger.info("⚠️  You may need to accept the security warning in your browser")
        return str(cert_file), str(key_file)
        
    except Exception as e:
        logger.error(f"❌ Failed to generate SSL certificate: {e}")
        logger.info("🔧 Falling back to HTTP (will show as 'not secure')")
        return None, None

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("\n👋 Voice assistant stopped.")