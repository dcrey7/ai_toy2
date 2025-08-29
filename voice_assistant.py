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
    """Handles WebSocket communication with simple 5-second recording."""
    def __init__(self, app: 'VoiceAssistant', websocket: WebSocket):
        super().__init__(app)
        self.websocket = websocket
        self.client_id = str(id(websocket))
        self.audio_buffer = []
        self.voice_mode_enabled = False
        self.recording_timer: Optional[asyncio.TimerHandle] = None
        self.RECORDING_DURATION = 5.0  # 5 seconds of recording
        self.is_recording = False

    async def _start_recording_timer(self):
        """Start 5-second recording after TTS finishes"""
        if self.recording_timer:
            self.recording_timer.cancel()
        
        # Wait a bit for TTS to finish playing, then start recording
        await asyncio.sleep(1.0)  
        
        if not self.voice_mode_enabled:
            return
            
        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 Starting 5-second recording...", "websocket": self.websocket})
        await self.app.comms_out_queue.put({"type": "vad_status", "status": "listening", "websocket": self.websocket})
        
        self.is_recording = True
        self.audio_buffer = []
        
        # Set timer to stop recording after 5 seconds
        loop = asyncio.get_running_loop()
        self.recording_timer = loop.call_later(self.RECORDING_DURATION, self._recording_timeout_callback)

    def _recording_timeout_callback(self):
        """Called when 5-second recording is complete"""
        asyncio.create_task(self._process_recorded_audio())

    async def _process_recorded_audio(self):
        """Process the 5 seconds of recorded audio"""
        try:
            self.is_recording = False
            
            if not self.audio_buffer:
                await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": "⚠️ No audio recorded in 5 seconds", "websocket": self.websocket})
                # Start another recording cycle
                asyncio.create_task(self._start_recording_timer())
                return
            
            buffer_count = len(self.audio_buffer)
            await self.app.comms_out_queue.put({"type": "vad_status", "status": "processing", "websocket": self.websocket})
            await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"🎙️ Recording complete! Processing {buffer_count} chunks", "websocket": self.websocket})
            
            # Send audio chunks directly to STT for processing
            await self.app.stt_in_queue.put({"audio_chunks": self.audio_buffer.copy(), "websocket": self.websocket})
            self.audio_buffer = []
            
        except Exception as e:
            await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ Recording processing error: {e}", "websocket": self.websocket})

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
                        greeting_text = "Hi! I'm your AI voice assistant. How can I help you?"
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 Voice mode enabled - sending greeting", "websocket": self.websocket})
                        # Send greeting to chat UI
                        await self.app.comms_out_queue.put({"type": "response", "text": greeting_text, "websocket": self.websocket})
                        # Send greeting to TTS and start recording after it finishes
                        await self.app.tts_in_queue.put({"text": greeting_text, "websocket": self.websocket, "use_fallback": False, "start_recording": True})
                    else:
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🔇 Voice mode disabled", "websocket": self.websocket})
                        if self.recording_timer: 
                            self.recording_timer.cancel()
                        self.audio_buffer = []
                        self.is_recording = False

                elif msg_type == 'audio_chunk' and self.voice_mode_enabled and self.is_recording:
                    if audio_b64 := message.get('data', ''):
                        audio_data = base64.b64decode(audio_b64)
                        self.audio_buffer.append(audio_data)
                        
                        # Simple logging every 20 chunks
                        if len(self.audio_buffer) % 20 == 0:
                            await self.app.comms_out_queue.put({"type": "log", "level": "debug", "message": f"📼 Recording: {len(self.audio_buffer)} chunks", "websocket": self.websocket})

                elif msg_type == 'start_next_recording':
                    # Message from TTS agent to start next recording cycle
                    if self.voice_mode_enabled:
                        asyncio.create_task(self._start_recording_timer())

                elif msg_type == 'text_message':
                    user_text = message.get("text")
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"💬 Text message received: '{user_text}'", "websocket": self.websocket})
                    await self.app.llm_in_queue.put({"text": user_text, "websocket": self.websocket, "source": "text"})

        except WebSocketDisconnect:
            logger.info(f"Client {self.client_id} disconnected.")
        finally:
            if self.recording_timer: self.recording_timer.cancel()
            self.app.active_clients.pop(self.websocket, None)

class STTAgent(Agent):
    async def run(self):
        while True:
            job = await self.app.stt_in_queue.get()
            ws, audio_bytes = job["websocket"], job["audio_bytes"]
            try:
                await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "🟡 Transcribing...", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"📝 STT Agent received {len(audio_bytes)} bytes of audio", "websocket": ws})
                
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🔊 Converted to {len(samples)} audio samples, sending to Kyutai STT", "websocket": ws})
                text = self.app.stt_service.transcribe(samples, sample_rate=16000)
                
                if text:
                    await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ STT transcription: '{text}'", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "transcript", "text": text, "is_final": True, "websocket": ws})
                    await self.app.llm_in_queue.put({"text": text, "websocket": ws, "source": "voice"})
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": "⚠️ STT returned empty transcription", "websocket": ws})
            except Exception as e:
                logger.error(f"STTAgent error: {e}")
                await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ STT Error: {e}", "websocket": ws})
            finally:
                await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "🟢 Ready", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "vad_status", "status": "listening", "websocket": ws})

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
                    await self.app.tts_in_queue.put({"text": response_text, "websocket": ws, "start_recording": True})
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
            start_recording = job.get("start_recording", False)
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
                    
                    # After TTS completes, trigger next recording cycle if requested
                    if start_recording:
                        ws_agent = self.app.active_clients.get(ws)
                        if ws_agent:
                            await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 TTS finished - starting recording cycle", "websocket": ws})
                            asyncio.create_task(ws_agent._start_recording_timer())
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