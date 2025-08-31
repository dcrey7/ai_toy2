"""
Persistent Metrics Export Service
Handles saving metrics to files with timestamps and structured logging
"""

import json
import csv
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
from dataclasses import asdict

from .metrics_service import metrics_collector, ResponseMetrics, SystemMetrics

logger = logging.getLogger(__name__)

class MetricsExportService:
    """Service for exporting metrics to persistent storage"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self.export_lock = threading.Lock()
        
        # Auto-export settings
        self.auto_export_enabled = True
        self.export_interval = 60  # Export every 60 seconds
        self.last_export_time = time.time()
        
        logger.info(f"📁 Metrics export service initialized - logs dir: {self.logs_dir}")
    
    def get_timestamp_filename(self, base_name: str, extension: str = "json") -> str:
        """Generate timestamped filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.{extension}"
    
    def export_response_metrics(self, responses: List[ResponseMetrics], filename: Optional[str] = None) -> str:
        """Export response metrics to JSON file"""
        if not filename:
            filename = self.get_timestamp_filename("response_metrics")
        
        filepath = self.logs_dir / filename
        
        with self.export_lock:
            try:
                # Convert ResponseMetrics to dict
                metrics_data = []
                for response in responses:
                    response_dict = asdict(response)
                    # Calculate derived metrics
                    derived = self._calculate_derived_metrics(response)
                    response_dict["derived_metrics"] = derived
                    metrics_data.append(response_dict)
                
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_responses": len(metrics_data),
                    "metrics": metrics_data
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"✅ Exported {len(responses)} response metrics to {filepath}")
                return str(filepath)
                
            except Exception as e:
                logger.error(f"❌ Failed to export response metrics: {e}")
                raise
    
    def export_system_metrics(self, system_metrics: List[SystemMetrics], filename: Optional[str] = None) -> str:
        """Export system metrics to CSV file"""
        if not filename:
            filename = self.get_timestamp_filename("system_metrics", "csv")
        
        filepath = self.logs_dir / filename
        
        with self.export_lock:
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        "timestamp", "datetime", "cpu_percent", "ram_percent", "ram_used_mb",
                        "gpu_utilization", "gpu_memory_used_mb", "gpu_memory_total_mb", 
                        "process_cpu_percent", "process_ram_mb"
                    ])
                    
                    # Data rows
                    for metric in system_metrics:
                        dt = datetime.fromtimestamp(metric.timestamp)
                        writer.writerow([
                            metric.timestamp, dt.isoformat(), metric.cpu_percent, 
                            metric.ram_percent, metric.ram_used_mb, metric.gpu_utilization,
                            metric.gpu_memory_used_mb, metric.gpu_memory_total_mb,
                            metric.process_cpu_percent, metric.process_ram_mb
                        ])
                
                logger.info(f"✅ Exported {len(system_metrics)} system metrics to {filepath}")
                return str(filepath)
                
            except Exception as e:
                logger.error(f"❌ Failed to export system metrics: {e}")
                raise
    
    def export_performance_summary(self, filename: Optional[str] = None) -> str:
        """Export comprehensive performance summary"""
        if not filename:
            filename = self.get_timestamp_filename("performance_summary")
        
        filepath = self.logs_dir / filename
        
        with self.export_lock:
            try:
                # Get all current metrics
                performance_data = metrics_collector.get_performance_summary()
                export_data = metrics_collector.export_metrics()
                
                # Add derived analysis
                analysis = self._generate_performance_analysis(export_data)
                
                summary = {
                    "export_timestamp": datetime.now().isoformat(),
                    "session_summary": performance_data,
                    "detailed_analysis": analysis,
                    "raw_metrics": export_data
                }
                
                with open(filepath, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                logger.info(f"✅ Exported performance summary to {filepath}")
                return str(filepath)
                
            except Exception as e:
                logger.error(f"❌ Failed to export performance summary: {e}")
                raise
    
    def export_session_log(self, session_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export complete session log with all metrics"""
        if not filename:
            filename = self.get_timestamp_filename("session_log")
        
        filepath = self.logs_dir / filename
        
        with self.export_lock:
            try:
                session_log = {
                    "session_start": session_data.get("session_start", time.time()),
                    "export_timestamp": datetime.now().isoformat(),
                    "duration_seconds": time.time() - session_data.get("session_start", time.time()),
                    **session_data
                }
                
                with open(filepath, 'w') as f:
                    json.dump(session_log, f, indent=2, default=str)
                
                logger.info(f"✅ Exported session log to {filepath}")
                return str(filepath)
                
            except Exception as e:
                logger.error(f"❌ Failed to export session log: {e}")
                raise
    
    def _calculate_derived_metrics(self, response: ResponseMetrics) -> Dict[str, Any]:
        """Calculate derived metrics for a response"""
        derived = {}
        
        # Voice-to-voice latency
        if response.vad_complete and response.tts_complete:
            derived["voice_to_voice_latency"] = response.tts_complete - response.vad_complete
        
        # Component latencies
        if response.stt_start and response.stt_complete:
            derived["stt_latency"] = response.stt_complete - response.stt_start
            
        if response.llm_start and response.llm_complete:
            derived["llm_latency"] = response.llm_complete - response.llm_start
            
        if response.tts_start and response.tts_complete:
            derived["tts_latency"] = response.tts_complete - response.tts_start
        
        # Time to first token
        if response.llm_start and response.llm_first_token:
            derived["llm_ttft"] = response.llm_first_token - response.llm_start
        
        # Tokens per second
        if response.tokens_generated and response.llm_start and response.llm_complete:
            generation_time = response.llm_complete - response.llm_start
            if generation_time > 0:
                derived["tokens_per_second"] = response.tokens_generated / generation_time
        
        # Performance targets
        targets = {
            "voice_to_voice_target": 0.8,
            "stt_target": 0.3,
            "llm_target": 0.3,
            "tts_target": 0.2
        }
        
        targets_met = {}
        for metric, target in targets.items():
            metric_key = metric.replace("_target", "_latency")
            if metric_key in derived:
                targets_met[metric_key + "_meets_target"] = derived[metric_key] <= target
        
        derived["performance_targets"] = targets_met
        
        return derived
    
    def _generate_performance_analysis(self, export_data: Dict) -> Dict[str, Any]:
        """Generate performance analysis from exported data"""
        analysis = {
            "session_duration": time.time() - export_data.get("session_start", time.time()),
            "total_responses": len(export_data.get("completed_responses", [])),
            "voice_responses": 0,
            "text_responses": 0,
            "avg_voice_latency": None,
            "avg_tokens_per_second": None,
            "error_rate": 0,
            "target_compliance": {}
        }
        
        completed_responses = export_data.get("completed_responses", [])
        
        if not completed_responses:
            return analysis
        
        # Count response types
        voice_latencies = []
        text_tps = []
        total_errors = 0
        
        for response_data in completed_responses:
            response = response_data.get("metrics", {})
            mode = response_data.get("mode", "unknown")
            
            if mode == "voice":
                analysis["voice_responses"] += 1
                if "voice_to_voice_latency" in response:
                    voice_latencies.append(response["voice_to_voice_latency"])
                    
            elif mode == "text":
                analysis["text_responses"] += 1
                if "tokens_per_second" in response:
                    text_tps.append(response["tokens_per_second"])
            
            # Count errors
            total_errors += response.get("errors", 0)
        
        # Calculate averages
        if voice_latencies:
            analysis["avg_voice_latency"] = sum(voice_latencies) / len(voice_latencies)
            analysis["p95_voice_latency"] = sorted(voice_latencies)[int(len(voice_latencies) * 0.95)]
            
        if text_tps:
            analysis["avg_tokens_per_second"] = sum(text_tps) / len(text_tps)
        
        # Error rate
        if completed_responses:
            analysis["error_rate"] = total_errors / len(completed_responses)
        
        # Target compliance
        if voice_latencies:
            target_met = sum(1 for lat in voice_latencies if lat <= 0.8) / len(voice_latencies)
            analysis["target_compliance"]["voice_to_voice_800ms"] = target_met
        
        return analysis
    
    def auto_export_if_needed(self):
        """Auto-export metrics if interval has passed"""
        if not self.auto_export_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_export_time >= self.export_interval:
            try:
                self.export_performance_summary()
                self.last_export_time = current_time
            except Exception as e:
                logger.error(f"Auto-export failed: {e}")
    
    def cleanup_old_files(self, days_to_keep: int = 7):
        """Clean up old log files"""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        try:
            for file_path in self.logs_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"🗑️ Cleaned up old log file: {file_path}")
                    
            for file_path in self.logs_dir.glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"🗑️ Cleaned up old log file: {file_path}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")

# Global export service instance
metrics_exporter = MetricsExportService()