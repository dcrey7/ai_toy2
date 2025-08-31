"""
Metrics API endpoints for accessing performance data via WebSocket
"""

import asyncio
import json
from typing import Dict, Any, Optional
import logging

from .metrics_service import metrics_collector

logger = logging.getLogger(__name__)

class MetricsAPI:
    """WebSocket API for accessing metrics data"""
    
    def __init__(self, app):
        self.app = app
        
    async def handle_metrics_request(self, message: Dict, websocket) -> Optional[Dict]:
        """Handle metrics-related WebSocket requests"""
        request_type = message.get("request_type")
        
        if request_type == "get_performance_summary":
            return await self.get_performance_summary(websocket)
        elif request_type == "get_conversation_stats":
            conversation_id = message.get("conversation_id")
            return await self.get_conversation_stats(conversation_id, websocket)
        elif request_type == "get_system_metrics":
            seconds = message.get("seconds", 60)
            return await self.get_system_metrics(seconds, websocket)
        elif request_type == "export_all_metrics":
            return await self.export_all_metrics(websocket)
        
        return None
    
    async def get_performance_summary(self, websocket) -> Dict:
        """Get overall performance summary"""
        summary = metrics_collector.get_performance_summary()
        return {
            "type": "metrics_response",
            "request_type": "performance_summary",
            "data": summary,
            "websocket": websocket
        }
    
    async def get_conversation_stats(self, conversation_id: str, websocket) -> Dict:
        """Get stats for a specific conversation"""
        stats = metrics_collector.get_conversation_stats(conversation_id)
        return {
            "type": "metrics_response", 
            "request_type": "conversation_stats",
            "conversation_id": conversation_id,
            "data": stats,
            "websocket": websocket
        }
    
    async def get_system_metrics(self, seconds: int, websocket) -> Dict:
        """Get recent system resource metrics"""
        metrics = metrics_collector.get_recent_system_metrics(seconds)
        
        # Convert to serializable format
        metrics_data = []
        for metric in metrics:
            metrics_data.append({
                "timestamp": metric.timestamp,
                "cpu_percent": metric.cpu_percent,
                "ram_percent": metric.ram_percent,
                "ram_used_mb": metric.ram_used_mb,
                "gpu_utilization": metric.gpu_utilization,
                "gpu_memory_used_mb": metric.gpu_memory_used_mb,
                "gpu_memory_total_mb": metric.gpu_memory_total_mb,
                "process_cpu_percent": metric.process_cpu_percent,
                "process_ram_mb": metric.process_ram_mb
            })
        
        return {
            "type": "metrics_response",
            "request_type": "system_metrics", 
            "data": {
                "metrics": metrics_data,
                "seconds": seconds
            },
            "websocket": websocket
        }
    
    async def export_all_metrics(self, websocket) -> Dict:
        """Export complete metrics dataset"""
        export_data = metrics_collector.export_metrics()
        return {
            "type": "metrics_response",
            "request_type": "export_all",
            "data": export_data,
            "websocket": websocket
        }
    
    async def send_live_metrics_update(self, websocket):
        """Send live metrics update to UI"""
        try:
            # Get recent system metrics (last 10 seconds)
            system_metrics = metrics_collector.get_recent_system_metrics(10)
            
            if system_metrics:
                latest_metric = system_metrics[-1]  # Most recent
                
                metrics_update = {
                    "type": "live_metrics",
                    "timestamp": latest_metric.timestamp,
                    "cpu_percent": latest_metric.cpu_percent,
                    "ram_percent": latest_metric.ram_percent, 
                    "gpu_utilization": latest_metric.gpu_utilization,
                    "gpu_memory_percent": (latest_metric.gpu_memory_used_mb / latest_metric.gpu_memory_total_mb * 100) if latest_metric.gpu_memory_total_mb > 0 else 0,
                    "websocket": websocket
                }
                
                await self.app.comms_out_queue.put(metrics_update)
                
        except Exception as e:
            logger.error(f"Error sending live metrics update: {e}")
    
    async def start_live_metrics_stream(self, websocket, interval: float = 2.0):
        """Start streaming live metrics to the UI"""
        try:
            while True:
                await self.send_live_metrics_update(websocket)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Live metrics stream cancelled")
        except Exception as e:
            logger.error(f"Live metrics stream error: {e}")

# Global metrics API instance
metrics_api = MetricsAPI(None)  # Will be set when VoiceAssistant initializes