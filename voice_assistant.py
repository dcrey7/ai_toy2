#!/usr/bin/env python3
"""
AI Voice Assistant - v2.2 (Streaming VAD & UI Logging)
"""

# Set CUDA environment variables BEFORE any torch imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Fix CUDA initialization issues
try:
    import torch
    # Clear CUDA cache and reset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Test CUDA availability
        device = torch.cuda.current_device()
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
except Exception as e:
    print(f"CUDA setup warning: {e}")

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
from services.tts_service import LocalTTSService, IsolatedTTSService
from services.ollama_service import OllamaService
from services.smart_turn_vad import SmartTurnVADService
from services.metrics_service import metrics_collector
from services.metrics_api import metrics_api
from services.metrics_export import metrics_exporter

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
                        
                        # Use existing VAD session if available, otherwise start new response
                        if hasattr(self, '_current_vad_session_id'):
                            response_id = self._current_vad_session_id
                            delattr(self, '_current_vad_session_id')  # Clear session
                        else:
                            response_id = metrics_collector.start_response(self.client_id, mode="voice")
                        
                        metrics_collector.mark_vad_complete(response_id)
                        
                        await self.app.comms_out_queue.put({
                            "type": "log",
                            "level": "success",
                            "message": f"🎯 Turn complete detected (confidence: {confidence:.3f}) [Response: {response_id}]",
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
                                "sample_rate": 16000,
                                "response_id": response_id  # Pass response_id for metrics
                            })
                            
                            # Clear buffer after sending to STT
                            self.vad_service.clear_buffer()
                        else:
                            logger.warning("⚠️ No audio buffer available for STT")
                            metrics_collector.add_error(response_id, "No audio buffer available")
                        
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
                    gpu_name = torch.cuda.get_device_name(0)
                    await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": f"🟢 {gpu_name}", "websocket": self.websocket})
                else:
                    await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": "⚪ CPU Only", "websocket": self.websocket})
            except Exception as e:
                await self.app.comms_out_queue.put({"type": "status", "component": "gpu", "status": "❌ GPU Error", "websocket": self.websocket})

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
                            
                            # Check if this is the start of speech (VAD start)
                            if not hasattr(self, '_current_vad_session_id'):
                                # Start new VAD session
                                self._current_vad_session_id = metrics_collector.start_response(self.client_id, mode="voice")
                                metrics_collector.mark_vad_start(self._current_vad_session_id)
                                await self.app.comms_out_queue.put({
                                    "type": "log", 
                                    "level": "info", 
                                    "message": f"🎤 VAD session started [Response: {self._current_vad_session_id}]",
                                    "websocket": self.websocket
                                })
                            
                            # Add to VAD service buffer
                            self.vad_service.add_pcm_chunk(pcm_array)
                            
                            # Update VAD status to show activity
                            await self.app.comms_out_queue.put({"type": "vad_status", "status": "speaking", "websocket": self.websocket})
                            
                    except Exception as e:
                        await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ PCM processing error: {e}", "websocket": self.websocket})
                

                elif msg_type == 'text_message':
                    user_text = message.get("text")
                    # Start metrics tracking for text response
                    response_id = metrics_collector.start_response(self.client_id, mode="text")
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"💬 Text message received: '{user_text}' [Response: {response_id}]", "websocket": self.websocket})
                    await self.app.llm_in_queue.put({"text": user_text, "websocket": self.websocket, "source": "text", "response_id": response_id})
                
                elif msg_type == 'metrics_request':
                    # Handle metrics API requests
                    response = await metrics_api.handle_metrics_request(message, self.websocket)
                    if response:
                        await self.app.comms_out_queue.put(response)
                
                elif msg_type == 'start_live_metrics':
                    # Start live metrics streaming
                    interval = message.get("interval", 2.0)
                    asyncio.create_task(metrics_api.start_live_metrics_stream(self.websocket, interval))
                    await self.app.comms_out_queue.put({
                        "type": "log", 
                        "level": "info", 
                        "message": f"📊 Started live metrics stream (interval: {interval}s)", 
                        "websocket": self.websocket
                    })
                
                elif msg_type == 'voice_timing_metrics':
                    # Handle client-side voice timing metrics
                    response_id = message.get("responseId")
                    voice_to_voice_latency = message.get("voiceToVoiceLatency")
                    
                    if response_id and voice_to_voice_latency is not None:
                        # Update metrics with client-side measurement
                        metrics_collector.update_client_timing(response_id, voice_to_voice_latency)
                        await self.app.comms_out_queue.put({
                            "type": "log", 
                            "level": "info", 
                            "message": f"📊 Client voice-to-voice timing: {voice_to_voice_latency:.3f}s [Response: {response_id}]", 
                            "websocket": self.websocket
                        })

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
            response_id = job.get("response_id")  # Get response_id for metrics
            
            try:
                # Mark STT start in metrics
                if response_id:
                    metrics_collector.mark_stt_start(response_id)
                
                await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "working", "websocket": ws})
                
                if audio_data is None or (isinstance(audio_data, list) and len(audio_data) == 0):
                    raise Exception("No audio data received")
                
                # Handle different audio formats
                if source == "smart_turn_vad":
                    # Direct PCM data from Smart Turn v2 VAD (numpy array)
                    if isinstance(audio_data, np.ndarray):
                        samples = audio_data.astype(np.float32)
                        await self.app.comms_out_queue.put({"type": "status", "component": "vadSystem", "status": "working", "websocket": ws})
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
                    # Mark STT completion in metrics
                    if response_id:
                        metrics_collector.mark_stt_complete(response_id, text)
                    
                    await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ STT transcription: '{text}'", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "status", "component": "stt", "status": "ready", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "status", "component": "vadSystem", "status": "ready", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "transcript", "text": text, "is_final": True, "websocket": ws})
                    await self.app.llm_in_queue.put({"text": text, "websocket": ws, "source": "voice", "response_id": response_id})
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "warning", "message": "⚠️ STT returned empty transcription - Smart Turn v2 continues listening", "websocket": ws})
                    if response_id:
                        metrics_collector.add_warning(response_id, "STT returned empty transcription")
                        
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
            response_id = job.get("response_id")  # Get response_id for metrics
            
            try:
                # Mark LLM start in metrics
                if response_id:
                    metrics_collector.mark_llm_start(response_id)
                
                await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "working", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🤖 LLM Agent processing: '{text}' (source: {source})", "websocket": ws})
                
                # Start streaming response
                full_response = ""
                speech_chunks = []  # Collect complete sentences for TTS
                
                await self.app.comms_out_queue.put({"type": "streaming_start", "websocket": ws})
                
                # Track first token timing
                first_token_received = False
                
                async for chunk in self.app.llm_service.generate_streaming_response(text, context=self.app.conversation_history):
                    if chunk["type"] == "sentence":
                        content = chunk["content"]
                        full_response += content + " "
                        speech_chunks.append(content)
                        
                        # Mark first token for metrics
                        if not first_token_received and response_id:
                            metrics_collector.mark_llm_first_token(response_id)
                            first_token_received = True
                        
                        # Send sentence immediately to frontend
                        await self.app.comms_out_queue.put({
                            "type": "streaming_chunk", 
                            "text": content, 
                            "websocket": ws
                        })
                        
                        # For voice mode, collect sentences for sequential TTS processing (disabled parallel to prevent overlap)
                        # if source == "voice" and len(content.strip()) > 15:  # Only meaningful sentences
                        #     await self.app.tts_in_queue.put({
                        #         "text": content, 
                        #         "websocket": ws, 
                        #         "is_streaming": True
                        #     })
                    
                    elif chunk["type"] == "partial":
                        content = chunk["content"]
                        full_response += content + " "
                        
                        # Send partial content for immediate display
                        await self.app.comms_out_queue.put({
                            "type": "streaming_chunk", 
                            "text": content, 
                            "websocket": ws
                        })
                    
                    elif chunk["type"] == "complete":
                        await self.app.comms_out_queue.put({"type": "streaming_complete", "websocket": ws})
                        break
                        
                    elif chunk["type"] == "error":
                        full_response = chunk["content"]
                        await self.app.comms_out_queue.put({
                            "type": "streaming_chunk", 
                            "text": full_response, 
                            "websocket": ws
                        })
                        await self.app.comms_out_queue.put({"type": "streaming_complete", "websocket": ws})
                        break
                
                # Mark LLM completion in metrics with proper token counting
                if response_id:
                    # More accurate token count estimation (approximation: 1 token ≈ 0.75 words)
                    word_count = len(full_response.split()) if full_response else 0
                    token_count = int(word_count * 1.33)  # Convert words to approximate tokens
                    
                    # Add text quality metrics
                    char_count = len(full_response.strip())
                    sentence_count = full_response.count('.') + full_response.count('!') + full_response.count('?')
                    
                    metrics_collector.mark_llm_complete(response_id, full_response.strip(), token_count)
                    
                    # Add quality metrics
                    metrics_collector.add_quality_metrics(response_id, {
                        "word_count": word_count,
                        "character_count": char_count,
                        "sentence_count": max(sentence_count, 1),  # At least 1
                        "avg_words_per_sentence": word_count / max(sentence_count, 1)
                    })
                
                # Update conversation history
                self.app.conversation_history.append({"role": "user", "content": text})
                self.app.conversation_history.append({"role": "assistant", "content": full_response.strip()})
                self.app.conversation_history = self.app.conversation_history[-20:]
                
                await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ LLM streaming complete: '{full_response[:100]}...'", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "ready", "websocket": ws})
                
                # Log completion and process TTS sequentially for voice mode
                if source != "voice":
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "📝 Text mode - streaming complete, skipping TTS synthesis", "websocket": ws})
                    # For text mode, complete the metrics tracking here
                    if response_id:
                        metrics = metrics_collector.complete_response(response_id)
                        if metrics:
                            await self.app.comms_out_queue.put({
                                "type": "log", 
                                "level": "info", 
                                "message": f"📊 Text Response Metrics: {metrics.get('tokens_per_second', 0):.1f} tokens/sec, {metrics.get('total_duration', 0):.2f}s total", 
                                "websocket": ws
                            })
                else:
                    # Send complete response to TTS as single chunk to avoid overlapping
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🎵 Sending complete response to TTS (sequential processing)", "websocket": ws})
                    await self.app.tts_in_queue.put({
                        "text": full_response.strip(), 
                        "websocket": ws, 
                        "is_streaming": False,
                        "response_id": response_id  # Pass response_id for TTS metrics
                    })
                    
            except Exception as e:
                logger.error(f"LLMAgent error: {e}")
                await self.app.comms_out_queue.put({"type": "log", "level": "error", "message": f"❌ LLM Error: {e}", "websocket": ws})
                await self.app.comms_out_queue.put({"type": "streaming_complete", "websocket": ws})
            finally:
                await self.app.comms_out_queue.put({"type": "status", "component": "llm", "status": "🟢 Ready", "websocket": ws})

class TTSAgent(Agent):
    async def run(self):
        while True:
            job = await self.app.tts_in_queue.get()
            ws, text = job["websocket"], job["text"]
            use_fallback = job.get("use_fallback", False)
            is_greeting = job.get("is_greeting", False)
            is_streaming = job.get("is_streaming", False)
            response_id = job.get("response_id")  # Get response_id for metrics
            
            try:
                # Mark TTS start in metrics
                if response_id:
                    metrics_collector.mark_tts_start(response_id)
                
                await self.app.comms_out_queue.put({"type": "status", "component": "tts", "status": "working", "websocket": ws})
                
                # Log different messages for streaming vs batch TTS
                if is_streaming:
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🎵 TTS Agent (streaming): '{text[:50]}...'", "websocket": ws})
                else:
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🎵 TTS Agent processing: '{text[:50]}...' (fallback={use_fallback})", "websocket": ws})
                
                # Sanitize text for TTS to avoid phonemizer errors with markdown
                sanitized_text = re.sub(r'[*`#_]', '', text)

                # Use isolated TTS service if available, otherwise fallback to direct service
                if hasattr(self.app, 'isolated_tts_service') and self.app.isolated_tts_service:
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🔒 Using isolated TTS process", "websocket": ws})
                    audio_data, sample_rate = await self.app.isolated_tts_service.synthesize(
                        sanitized_text, 
                        voice="af_heart", 
                        force_fallback=use_fallback
                    )
                else:
                    # Fallback to direct TTS service
                    engine_name = "espeak" if use_fallback else self.app.tts_service.primary_engine
                    await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": f"🔊 Using direct TTS engine: {engine_name}", "websocket": ws})
                    audio_data, sample_rate = self.app.tts_service.synthesize(sanitized_text, voice="af_heart", force_fallback=use_fallback)
                
                if audio_data is not None and len(audio_data) > 0:
                    duration = len(audio_data) / sample_rate
                    
                    # Mark TTS completion in metrics
                    if response_id:
                        metrics_collector.mark_tts_complete(response_id, duration)
                        # Complete the voice response metrics tracking
                        metrics = metrics_collector.complete_response(response_id)
                        if metrics and not is_streaming:  # Only show full metrics for non-streaming TTS
                            await self.app.comms_out_queue.put({
                                "type": "log", 
                                "level": "info", 
                                "message": f"📊 Voice Response Metrics: {metrics.get('voice_to_voice_latency', 0):.2f}s total, STT: {metrics.get('stt_latency', 0):.2f}s, LLM: {metrics.get('llm_latency', 0):.2f}s, TTS: {metrics.get('tts_latency', 0):.2f}s", 
                                "websocket": ws
                            })
                    
                    await self.app.comms_out_queue.put({"type": "log", "level": "success", "message": f"✅ TTS generated {len(audio_data)} samples ({duration:.1f}s) at {sample_rate}Hz", "websocket": ws})
                    await self.app.comms_out_queue.put({"type": "status", "component": "tts", "status": "ready", "websocket": ws})
                    
                    # Convert to int16 and encode as base64
                    audio_b64 = base64.b64encode((audio_data * 32767).astype(np.int16).tobytes()).decode('utf-8')
                    await self.app.comms_out_queue.put({
                        "type": "tts_audio", 
                        "audio": audio_b64, 
                        "sample_rate": sample_rate, 
                        "text": text, 
                        "is_streaming": is_streaming,
                        "websocket": ws
                    })
                    
                    # Handle different post-TTS behaviors
                    if is_greeting:
                        # After greeting TTS finishes, start VAD listening with a brief delay
                        await asyncio.sleep(1.0)  # Wait 1 second after greeting ends
                        await self.app.comms_out_queue.put({"type": "vad_status", "status": "inactive", "websocket": ws})
                        await self.app.comms_out_queue.put({"type": "log", "level": "info", "message": "🎤 Greeting finished - Smart Turn v2 VAD now listening for your voice", "websocket": ws})
                    elif is_streaming:
                        # For streaming TTS, just log completion
                        await self.app.comms_out_queue.put({"type": "log", "level": "debug", "message": f"🎵 Streaming TTS chunk completed: '{text[:30]}...'", "websocket": ws})
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
        self.isolated_tts_service: Optional[IsolatedTTSService] = None
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
        logger.info("🚀 Initializing AI services with GPU optimization...")
        
        # Start system resource monitoring
        await metrics_collector.start_system_monitoring(interval=1.0)
        logger.info("📊 System metrics monitoring started")
        
        # Start auto-export task (every 5 minutes)
        asyncio.create_task(self._auto_export_metrics_loop())
        logger.info("📁 Auto-export metrics task started")
        
        # Initialize metrics API with this app instance
        metrics_api.app = self
        
        # Configure multiprocessing for CUDA compatibility
        import torch.multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
                logger.info("✅ Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError as e:
            logger.info(f"Multiprocessing context already set: {e}")
        
        # Initialize CUDA context properly with error handling
        try:
            # Force CUDA reinitialization
            torch.cuda.is_available()  # This call initializes CUDA
            
            if torch.cuda.is_available():
                # Try to get device info - this will fail if CUDA is misconfigured
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    torch.cuda.empty_cache()
                    logger.info(f"✅ CUDA initialized - GPU: {gpu_name}")
                else:
                    logger.warning("⚠️ No CUDA devices found")
            else:
                logger.warning("⚠️ CUDA not available, models will run on CPU")
        except Exception as cuda_error:
            logger.warning(f"⚠️ CUDA error: {cuda_error}. Falling back to CPU.")
            # Don't change environment variables after program start - this causes CUDA errors
            # Instead, let services handle CPU fallback internally
        
        # Initialize services sequentially to avoid CUDA conflicts
        try:
            # 1. Initialize LLM service (Ollama) first - most important for GPU
            logger.info("📡 Initializing LLM service (Ollama)...")
            llm_config = self.config['models']['llm']
            self.llm_service = OllamaService(base_url=llm_config['base_url'], model_name=llm_config['model_name'])
            
            # 2. Initialize STT service
            logger.info("🎤 Initializing STT service...")
            stt_config = self.config['models']['stt']
            self.stt_service = KyutaiSTTService(
                model_name=stt_config['model_name'],
                device=stt_config['device']
            )
            
            # Clear cache after STT loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 3. Initialize direct TTS service (lighter load first)
            logger.info("🔊 Initializing TTS service...")
            tts_config = self.config['models']['tts']
            self.tts_service = LocalTTSService(primary_engine=tts_config['primary'], device=tts_config.get('device', 'cuda'))
            
            # Clear cache after TTS loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 4. Initialize isolated TTS service for better CUDA memory management
            try:
                logger.info("🔄 Initializing isolated TTS service...")
                self.isolated_tts_service = IsolatedTTSService()
                logger.info("✅ Isolated TTS service initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Isolated TTS service failed to initialize: {e}")
                logger.info("🔄 Falling back to direct TTS service")
                self.isolated_tts_service = None
                
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

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
        # Export final metrics before cleanup
        try:
            logger.info("📊 Exporting final session metrics...")
            export_path = metrics_exporter.export_performance_summary()
            logger.info(f"✅ Final metrics exported to: {export_path}")
        except Exception as e:
            logger.error(f"❌ Failed to export final metrics: {e}")
        
        # Stop system monitoring
        await metrics_collector.stop_system_monitoring()
        logger.info("📊 System metrics monitoring stopped")
        
        for agent in self.agents: await agent.stop()
        if self.stt_service: self.stt_service.cleanup()
        if self.tts_service: self.tts_service.cleanup()
        if self.isolated_tts_service: self.isolated_tts_service.cleanup()
        if self.llm_service: await self.llm_service.cleanup()
    
    async def _auto_export_metrics_loop(self):
        """Background task to auto-export metrics every 5 minutes"""
        export_interval = 300  # 5 minutes
        
        while True:
            try:
                await asyncio.sleep(export_interval)
                
                # Export performance summary
                export_path = metrics_exporter.export_performance_summary()
                logger.info(f"📁 Auto-exported metrics to: {export_path}")
                
                # Clean up old files (keep 7 days)
                metrics_exporter.cleanup_old_files(days_to_keep=7)
                
            except asyncio.CancelledError:
                logger.info("📁 Auto-export task cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Auto-export error: {e}")
                # Continue the loop despite errors

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