"""
Timing decorators for automatic performance instrumentation
"""

import time
import functools
import asyncio
import logging
from typing import Callable, Any, Dict, Optional
from .metrics_service import metrics_collector

logger = logging.getLogger(__name__)

def time_sync_method(component: str, operation: str):
    """
    Decorator to time synchronous methods and report to metrics
    
    Args:
        component: Component name (e.g., "stt", "llm", "tts")
        operation: Operation name (e.g., "transcribe", "generate", "synthesize")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Log timing
                logger.info(f"⏱️ {component}.{operation}: {duration:.3f}s")
                
                # TODO: Report to metrics collector when response_id is available
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"❌ {component}.{operation} failed after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator

def time_async_method(component: str, operation: str):
    """
    Decorator to time asynchronous methods and report to metrics
    
    Args:
        component: Component name (e.g., "stt", "llm", "tts") 
        operation: Operation name (e.g., "transcribe", "generate", "synthesize")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                
                # Log timing
                logger.info(f"⏱️ {component}.{operation}: {duration:.3f}s")
                
                # TODO: Report to metrics collector when response_id is available
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(f"❌ {component}.{operation} failed after {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator

def track_response_stage(stage: str):
    """
    Decorator to track specific response pipeline stages
    
    Args:
        stage: Stage name ("vad_start", "stt_start", "stt_complete", etc.)
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Try to find response_id in args/kwargs
                response_id = _extract_response_id(args, kwargs)
                
                if response_id:
                    _mark_stage(response_id, stage)
                
                result = await func(*args, **kwargs)
                
                # Mark completion stages
                if stage.endswith("_start"):
                    completion_stage = stage.replace("_start", "_complete")
                    if response_id:
                        _mark_stage(response_id, completion_stage)
                
                return result
            return async_wrapper
        else:
            @functools.wraps(func) 
            def sync_wrapper(*args, **kwargs):
                # Try to find response_id in args/kwargs
                response_id = _extract_response_id(args, kwargs)
                
                if response_id:
                    _mark_stage(response_id, stage)
                
                result = func(*args, **kwargs)
                
                # Mark completion stages
                if stage.endswith("_start"):
                    completion_stage = stage.replace("_start", "_complete")
                    if response_id:
                        _mark_stage(response_id, completion_stage)
                
                return result
            return sync_wrapper
    return decorator

def _extract_response_id(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract response_id from function arguments"""
    # Look for response_id in kwargs
    if "response_id" in kwargs:
        return kwargs["response_id"]
    
    # Look for response_id in args (assuming it's often the first arg after self)
    for arg in args[1:]:  # Skip 'self' 
        if isinstance(arg, str) and "_" in arg and arg.count("_") >= 2:
            # Looks like a response_id format: conversation_timestamp
            return arg
    
    return None

def _mark_stage(response_id: str, stage: str):
    """Mark a specific stage in the metrics collector"""
    stage_map = {
        "vad_start": metrics_collector.mark_vad_start,
        "vad_complete": metrics_collector.mark_vad_complete,
        "stt_start": metrics_collector.mark_stt_start,
        "stt_complete": metrics_collector.mark_stt_complete,
        "llm_start": metrics_collector.mark_llm_start,
        "llm_first_token": metrics_collector.mark_llm_first_token,
        "llm_complete": metrics_collector.mark_llm_complete,
        "tts_start": metrics_collector.mark_tts_start,
        "tts_complete": metrics_collector.mark_tts_complete,
        "streaming_start": metrics_collector.mark_streaming_start,
    }
    
    if stage in stage_map:
        try:
            stage_map[stage](response_id)
        except Exception as e:
            logger.error(f"Failed to mark stage {stage} for {response_id}: {e}")

# Convenience decorators for common service operations
def time_stt_transcribe(func):
    """Decorator specifically for STT transcription methods"""
    return time_async_method("stt", "transcribe")(func)

def time_llm_generate(func):
    """Decorator specifically for LLM generation methods"""
    return time_async_method("llm", "generate")(func)

def time_tts_synthesize(func):
    """Decorator specifically for TTS synthesis methods"""
    return time_async_method("tts", "synthesize")(func)

def time_vad_detect(func):
    """Decorator specifically for VAD detection methods"""
    return time_sync_method("vad", "detect")(func)