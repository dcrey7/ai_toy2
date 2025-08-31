"""
Comprehensive Metrics Collection Service for AI Voice Assistant
Tracks performance, latency, resource usage, and conversation quality
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ResponseMetrics:
    """Metrics for a single response cycle"""
    conversation_id: str
    response_id: str
    mode: str  # "text" or "voice"
    start_time: float
    end_time: Optional[float] = None
    
    # Voice-specific timings (voice-to-voice latency breakdown)
    vad_start: Optional[float] = None
    vad_complete: Optional[float] = None
    stt_start: Optional[float] = None
    stt_complete: Optional[float] = None
    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_complete: Optional[float] = None
    tts_start: Optional[float] = None
    tts_complete: Optional[float] = None
    
    # Text-specific metrics
    tokens_generated: Optional[int] = None
    streaming_started: Optional[float] = None
    
    # Quality metrics
    input_text: str = ""
    output_text: str = ""
    audio_duration: Optional[float] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Client-side measurements
    client_voice_to_voice_latency: Optional[float] = None
    
    # Quality metrics
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class SystemMetrics:
    """System resource usage metrics"""
    timestamp: float
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    gpu_utilization: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    process_cpu_percent: float
    process_ram_mb: float

class MetricsCollector:
    """Centralized metrics collection and analysis service"""
    
    def __init__(self):
        self.current_responses: Dict[str, ResponseMetrics] = {}
        self.completed_responses: deque = deque(maxlen=1000)  # Keep last 1000 responses
        self.system_metrics: deque = deque(maxlen=300)  # Keep last 5 minutes at 1s intervals
        
        # Real-time tracking
        self.active_conversations: Dict[str, Dict] = {}
        self.session_start_time = time.time()
        
        # Locks for thread safety
        self.metrics_lock = threading.Lock()
        
        # Resource monitoring
        self.system_monitor_active = False
        self.system_monitor_task = None
        
        # Performance targets (from SOTA voice agents blog)
        self.performance_targets = {
            "voice_to_voice_latency": 0.8,  # 800ms target
            "stt_latency": 0.3,  # 300ms STT
            "llm_latency": 0.3,  # 300ms LLM  
            "tts_latency": 0.2,  # 200ms TTS
            "tokens_per_second": 50,  # Target TPS for text generation
        }
        
        logger.info("🎯 Metrics service initialized with SOTA performance targets")

    def start_response(self, conversation_id: str, mode: str = "voice") -> str:
        """Start tracking a new response cycle"""
        response_id = f"{conversation_id}_{int(time.time() * 1000)}"
        
        with self.metrics_lock:
            self.current_responses[response_id] = ResponseMetrics(
                conversation_id=conversation_id,
                response_id=response_id,
                mode=mode,
                start_time=time.time()
            )
            
            # Initialize conversation tracking if new
            if conversation_id not in self.active_conversations:
                self.active_conversations[conversation_id] = {
                    "start_time": time.time(),
                    "response_count": 0,
                    "total_voice_latency": 0,
                    "total_text_tokens": 0,
                    "errors": 0
                }
                
        logger.info(f"📊 Started tracking {mode} response: {response_id}")
        return response_id
    
    def mark_vad_start(self, response_id: str):
        """Mark when VAD processing starts"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].vad_start = time.time()
    
    def mark_vad_complete(self, response_id: str):
        """Mark when VAD completes turn detection"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].vad_complete = time.time()
    
    def mark_stt_start(self, response_id: str):
        """Mark when STT processing starts"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].stt_start = time.time()
    
    def mark_stt_complete(self, response_id: str, transcribed_text: str = ""):
        """Mark when STT processing completes"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                response = self.current_responses[response_id]
                response.stt_complete = time.time()
                response.input_text = transcribed_text
    
    def mark_llm_start(self, response_id: str):
        """Mark when LLM processing starts"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].llm_start = time.time()
    
    def mark_llm_first_token(self, response_id: str):
        """Mark when LLM generates first token (TTFT - Time To First Token)"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].llm_first_token = time.time()
    
    def mark_llm_complete(self, response_id: str, generated_text: str = "", token_count: int = 0):
        """Mark when LLM completes generation"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                response = self.current_responses[response_id]
                response.llm_complete = time.time()
                response.output_text = generated_text
                response.tokens_generated = token_count
    
    def mark_tts_start(self, response_id: str):
        """Mark when TTS processing starts"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].tts_start = time.time()
    
    def mark_tts_complete(self, response_id: str, audio_duration: float = 0):
        """Mark when TTS processing completes"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                response = self.current_responses[response_id]
                response.tts_complete = time.time()
                response.audio_duration = audio_duration
    
    def mark_streaming_start(self, response_id: str):
        """Mark when text streaming starts (for text mode)"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].streaming_started = time.time()
    
    def complete_response(self, response_id: str) -> Optional[Dict]:
        """Complete response tracking and return metrics"""
        with self.metrics_lock:
            if response_id not in self.current_responses:
                return None
                
            response = self.current_responses.pop(response_id)
            response.end_time = time.time()
            
            # Calculate derived metrics
            metrics = self._calculate_response_metrics(response)
            
            # Update conversation tracking
            conv_id = response.conversation_id
            if conv_id in self.active_conversations:
                conv = self.active_conversations[conv_id]
                conv["response_count"] += 1
                
                if response.mode == "voice" and metrics.get("voice_to_voice_latency"):
                    conv["total_voice_latency"] += metrics["voice_to_voice_latency"]
                elif response.mode == "text" and response.tokens_generated:
                    conv["total_text_tokens"] += response.tokens_generated
            
            # Store completed response
            self.completed_responses.append(response)
            
            logger.info(f"✅ Completed {response.mode} response: {response_id}")
            logger.info(f"📈 Metrics: {json.dumps(metrics, indent=2)}")
            
            return metrics
    
    def add_error(self, response_id: str, error_msg: str):
        """Add error to response tracking"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].errors.append(error_msg)
                
                # Update conversation error count
                conv_id = self.current_responses[response_id].conversation_id
                if conv_id in self.active_conversations:
                    self.active_conversations[conv_id]["errors"] += 1
    
    def add_warning(self, response_id: str, warning_msg: str):
        """Add warning to response tracking"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].warnings.append(warning_msg)
    
    def update_client_timing(self, response_id: str, client_voice_to_voice_latency: float):
        """Update response with client-side voice-to-voice timing measurement"""
        with self.metrics_lock:
            # Check current responses first
            if response_id in self.current_responses:
                response = self.current_responses[response_id]
                response.client_voice_to_voice_latency = client_voice_to_voice_latency
                logger.info(f"📊 Updated current response {response_id} with client timing: {client_voice_to_voice_latency:.3f}s")
                return
            
            # Check completed responses
            for response in self.completed_responses:
                if response.response_id == response_id:
                    response.client_voice_to_voice_latency = client_voice_to_voice_latency
                    logger.info(f"📊 Updated completed response {response_id} with client timing: {client_voice_to_voice_latency:.3f}s")
                    return
                    
            logger.warning(f"⚠️ Could not find response {response_id} to update with client timing")
    
    def add_quality_metrics(self, response_id: str, quality_data: Dict[str, Any]):
        """Add quality metrics to response tracking"""
        with self.metrics_lock:
            if response_id in self.current_responses:
                self.current_responses[response_id].quality_metrics.update(quality_data)
                logger.debug(f"📊 Added quality metrics to response {response_id}: {quality_data}")
            else:
                logger.warning(f"⚠️ Could not find response {response_id} to add quality metrics")
    
    def _calculate_response_metrics(self, response: ResponseMetrics) -> Dict:
        """Calculate derived metrics from response data"""
        metrics = {
            "response_id": response.response_id,
            "conversation_id": response.conversation_id,
            "mode": response.mode,
            "total_duration": response.end_time - response.start_time if response.end_time else None,
            "errors": len(response.errors),
            "warnings": len(response.warnings)
        }
        
        if response.mode == "voice":
            # Voice-to-voice latency breakdown
            if response.vad_start and response.vad_complete:
                metrics["vad_latency"] = response.vad_complete - response.vad_start
                
            if response.stt_start and response.stt_complete:
                metrics["stt_latency"] = response.stt_complete - response.stt_start
                
            if response.llm_start and response.llm_complete:
                metrics["llm_latency"] = response.llm_complete - response.llm_start
                
            if response.llm_start and response.llm_first_token:
                metrics["llm_ttft"] = response.llm_first_token - response.llm_start
                
            if response.tts_start and response.tts_complete:
                metrics["tts_latency"] = response.tts_complete - response.tts_start
                
            # Total voice-to-voice latency (from VAD detection to TTS complete)
            if response.vad_complete and response.tts_complete:
                metrics["voice_to_voice_latency"] = response.tts_complete - response.vad_complete
                
            # Audio metrics
            if response.audio_duration:
                metrics["audio_duration"] = response.audio_duration
                
        elif response.mode == "text":
            # Text chat metrics
            if response.tokens_generated and response.llm_start and response.llm_complete:
                generation_time = response.llm_complete - response.llm_start
                metrics["tokens_per_second"] = response.tokens_generated / generation_time
                
            if response.streaming_started and response.start_time:
                metrics["time_to_first_token"] = response.streaming_started - response.start_time
                
            if response.tokens_generated:
                metrics["total_tokens"] = response.tokens_generated
        
        # Performance vs targets
        targets_met = {}
        for metric, target in self.performance_targets.items():
            if metric in metrics and metrics[metric] is not None:
                targets_met[metric] = metrics[metric] <= target
                
        metrics["targets_met"] = targets_met
        
        return metrics
    
    async def start_system_monitoring(self, interval: float = 1.0):
        """Start background system resource monitoring"""
        if self.system_monitor_active:
            return
            
        self.system_monitor_active = True
        self.system_monitor_task = asyncio.create_task(self._system_monitor_loop(interval))
        logger.info("🖥️ Started system resource monitoring")
    
    async def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.system_monitor_active = False
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🖥️ Stopped system resource monitoring")
    
    async def _system_monitor_loop(self, interval: float):
        """Background loop for system monitoring"""
        process = psutil.Process()
        
        while self.system_monitor_active:
            try:
                # System-wide metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # Process-specific metrics
                process_cpu = process.cpu_percent()
                process_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # GPU metrics (if available)
                gpu_util = 0
                gpu_memory_used = 0
                gpu_memory_total = 0
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Get GPU utilization (simplified)
                        gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
                        gpu_util = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
                except Exception as e:
                    logger.debug(f"GPU metrics error: {e}")
                
                # Store metrics
                system_metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    ram_percent=memory.percent,
                    ram_used_mb=memory.used / 1024 / 1024,
                    gpu_utilization=gpu_util,
                    gpu_memory_used_mb=gpu_memory_used,
                    gpu_memory_total_mb=gpu_memory_total,
                    process_cpu_percent=process_cpu,
                    process_ram_mb=process_memory
                )
                
                with self.metrics_lock:
                    self.system_metrics.append(system_metrics)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def get_conversation_stats(self, conversation_id: str) -> Dict:
        """Get statistics for a specific conversation"""
        with self.metrics_lock:
            if conversation_id not in self.active_conversations:
                return {}
                
            conv = self.active_conversations[conversation_id]
            
            # Calculate averages
            avg_voice_latency = None
            if conv["response_count"] > 0 and conv["total_voice_latency"] > 0:
                avg_voice_latency = conv["total_voice_latency"] / conv["response_count"]
            
            return {
                "conversation_id": conversation_id,
                "duration": time.time() - conv["start_time"],
                "response_count": conv["response_count"],
                "average_voice_latency": avg_voice_latency,
                "total_tokens": conv["total_text_tokens"],
                "error_count": conv["errors"],
                "error_rate": conv["errors"] / max(conv["response_count"], 1)
            }
    
    def get_recent_system_metrics(self, seconds: int = 60) -> List[SystemMetrics]:
        """Get recent system metrics for the last N seconds"""
        cutoff_time = time.time() - seconds
        
        with self.metrics_lock:
            return [m for m in self.system_metrics if m.timestamp > cutoff_time]
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        with self.metrics_lock:
            recent_responses = list(self.completed_responses)[-50:]  # Last 50 responses
            
        if not recent_responses:
            return {"message": "No completed responses yet"}
        
        # Voice metrics
        voice_responses = [r for r in recent_responses if r.mode == "voice"]
        text_responses = [r for r in recent_responses if r.mode == "text"]
        
        summary = {
            "total_responses": len(recent_responses),
            "voice_responses": len(voice_responses),
            "text_responses": len(text_responses),
            "session_duration": time.time() - self.session_start_time,
        }
        
        # Voice performance
        if voice_responses:
            voice_latencies = []
            for r in voice_responses:
                if r.vad_complete and r.tts_complete:
                    voice_latencies.append(r.tts_complete - r.vad_complete)
            
            if voice_latencies:
                summary["voice_metrics"] = {
                    "avg_voice_to_voice_latency": np.mean(voice_latencies),
                    "p95_voice_to_voice_latency": np.percentile(voice_latencies, 95),
                    "target_met_rate": sum(1 for lat in voice_latencies if lat <= 0.8) / len(voice_latencies)
                }
        
        # Text performance  
        if text_responses:
            tps_values = []
            for r in text_responses:
                if r.tokens_generated and r.llm_start and r.llm_complete:
                    generation_time = r.llm_complete - r.llm_start
                    if generation_time > 0:
                        tps_values.append(r.tokens_generated / generation_time)
            
            if tps_values:
                summary["text_metrics"] = {
                    "avg_tokens_per_second": np.mean(tps_values),
                    "min_tokens_per_second": np.min(tps_values),
                    "max_tokens_per_second": np.max(tps_values)
                }
        
        # Component-level performance breakdown
        if recent_responses:
            summary["component_performance"] = self._calculate_component_performance(recent_responses)
        
        return summary
    
    def export_metrics(self) -> Dict:
        """Export all metrics for analysis"""
        with self.metrics_lock:
            return {
                "session_start": self.session_start_time,
                "current_responses": len(self.current_responses),
                "completed_responses": [
                    {
                        "response_id": r.response_id,
                        "conversation_id": r.conversation_id, 
                        "mode": r.mode,
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                        "metrics": self._calculate_response_metrics(r)
                    }
                    for r in list(self.completed_responses)
                ],
                "active_conversations": dict(self.active_conversations),
                "system_metrics": [
                    {
                        "timestamp": m.timestamp,
                        "cpu_percent": m.cpu_percent,
                        "ram_percent": m.ram_percent,
                        "gpu_utilization": m.gpu_utilization,
                        "gpu_memory_used_mb": m.gpu_memory_used_mb
                    }
                    for m in list(self.system_metrics)
                ]
            }
    
    def _calculate_component_performance(self, responses: List[ResponseMetrics]) -> Dict:
        """Calculate per-component performance statistics"""
        component_stats = {
            "vad": {"latencies": [], "success_rate": 0, "avg_latency": 0},
            "stt": {"latencies": [], "success_rate": 0, "avg_latency": 0, "avg_audio_duration": 0},
            "llm": {"latencies": [], "ttft_latencies": [], "success_rate": 0, "avg_tokens_per_second": 0},
            "tts": {"latencies": [], "success_rate": 0, "avg_latency": 0, "avg_audio_duration": 0}
        }
        
        total_responses = len(responses)
        
        for response in responses:
            # VAD component
            if response.vad_start and response.vad_complete:
                vad_latency = response.vad_complete - response.vad_start
                component_stats["vad"]["latencies"].append(vad_latency)
            
            # STT component
            if response.stt_start and response.stt_complete:
                stt_latency = response.stt_complete - response.stt_start
                component_stats["stt"]["latencies"].append(stt_latency)
            
            # LLM component
            if response.llm_start and response.llm_complete:
                llm_latency = response.llm_complete - response.llm_start
                component_stats["llm"]["latencies"].append(llm_latency)
                
                # TTFT
                if response.llm_first_token:
                    ttft = response.llm_first_token - response.llm_start
                    component_stats["llm"]["ttft_latencies"].append(ttft)
            
            # TTS component  
            if response.tts_start and response.tts_complete:
                tts_latency = response.tts_complete - response.tts_start
                component_stats["tts"]["latencies"].append(tts_latency)
                
                if response.audio_duration:
                    component_stats["tts"]["avg_audio_duration"] = response.audio_duration
        
        # Calculate statistics
        for component, stats in component_stats.items():
            if stats["latencies"]:
                stats["avg_latency"] = np.mean(stats["latencies"])
                stats["p95_latency"] = np.percentile(stats["latencies"], 95)
                stats["success_rate"] = len(stats["latencies"]) / total_responses
                stats["sample_count"] = len(stats["latencies"])
                
                # Component-specific metrics
                if component == "llm" and stats["ttft_latencies"]:
                    stats["avg_ttft"] = np.mean(stats["ttft_latencies"])
                    stats["p95_ttft"] = np.percentile(stats["ttft_latencies"], 95)
                
                # Remove raw latency arrays to reduce size
                del stats["latencies"]
                if "ttft_latencies" in stats:
                    del stats["ttft_latencies"]
        
        return component_stats

# Global metrics collector instance
metrics_collector = MetricsCollector()