
"""
Ollama Service for LLM and Vision
Uses your existing Gemma 3B and Mistral 7B models
"""

import aiohttp
import asyncio
import json
import logging
import base64
import io
import re
from PIL import Image
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download both punkt and punkt_tab for compatibility
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    nltk = None
except Exception as e:
    print(f"NLTK download error: {e}")
    nltk = None

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, base_url="http://localhost:11434", model_name="gemma:3b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def generate_response(self, message, context=None, system_prompt=None):
        """
        Generate text response from LLM
        
        Args:
            message (str): User message
            context (list): Previous conversation context
            system_prompt (str): System prompt
            
        Returns:
            str: Generated response
        """
        try:
            session = await self._get_session()
            
            # Build messages array
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            if context:
                messages.extend(context)
                
            messages.append({"role": "user", "content": message})
            
            # Request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 150  # Keep responses concise for voice
                }
            }
            
            logger.debug(f"Sending request to Ollama: {message}")
            
            async with session.post(f"{self.base_url}/api/chat", 
                                  json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    reply = data.get("message", {}).get("content", "")
                    logger.debug(f"Ollama response: {reply}")
                    return reply.strip()
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    return "I'm having trouble thinking right now. Could you try again?"
                    
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return "I'm experiencing technical difficulties. Please try again."
    
    def _split_into_sentences(self, text):
        """Split text into sentences using NLTK or fallback regex"""
        if nltk:
            try:
                sentences = sent_tokenize(text)
                # Filter out very short sentences (< 10 chars)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}, using fallback")
        
        # Fallback regex-based sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
    
    async def generate_streaming_response(self, message, context=None, system_prompt=None):
        """
        Generate streaming text response from LLM with sentence-level chunks
        
        Args:
            message (str): User message
            context (list): Previous conversation context
            system_prompt (str): System prompt
            
        Yields:
            dict: {"type": "sentence", "content": str} or {"type": "complete"}
        """
        try:
            session = await self._get_session()
            
            # Build messages array
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            if context:
                messages.extend(context)
                
            messages.append({"role": "user", "content": message})
            
            # Request payload with streaming enabled
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200  # Slightly higher for streaming
                }
            }
            
            logger.debug(f"Starting streaming request to Ollama: {message}")
            
            async with session.post(f"{self.base_url}/api/chat", 
                                  json=payload) as response:
                if response.status == 200:
                    # Accumulate tokens to form complete sentences
                    accumulated_text = ""
                    sentence_buffer = ""
                    
                    async for line in response.content:
                        if line:
                            try:
                                # Each line should be a JSON object
                                line_str = line.decode('utf-8').strip()
                                if not line_str:
                                    continue
                                    
                                data = json.loads(line_str)
                                
                                # Check if streaming is complete
                                if data.get("done", False):
                                    # Send any remaining text
                                    if sentence_buffer.strip():
                                        yield {"type": "sentence", "content": sentence_buffer.strip()}
                                    yield {"type": "complete"}
                                    break
                                
                                # Get the delta content
                                delta_content = data.get("message", {}).get("content", "")
                                if not delta_content:
                                    continue
                                
                                sentence_buffer += delta_content
                                
                                # Check for sentence boundaries
                                if any(punct in delta_content for punct in ['.', '!', '?']):
                                    # Split buffer into sentences
                                    sentences = self._split_into_sentences(sentence_buffer)
                                    
                                    # Send complete sentences
                                    for sentence in sentences[:-1]:  # All but the last (potentially incomplete)
                                        if sentence.strip():
                                            yield {"type": "sentence", "content": sentence.strip()}
                                    
                                    # Keep the last potentially incomplete sentence
                                    sentence_buffer = sentences[-1] if sentences else ""
                                
                                # Also yield on comma or other natural breaks for smoother streaming
                                elif len(sentence_buffer) > 50 and any(punct in delta_content for punct in [',', ';', ':']):
                                    # Find the last punctuation for a good break point
                                    break_point = max(
                                        sentence_buffer.rfind(','),
                                        sentence_buffer.rfind(';'),
                                        sentence_buffer.rfind(':')
                                    )
                                    if break_point > 20:  # Ensure meaningful chunk
                                        chunk = sentence_buffer[:break_point+1].strip()
                                        if chunk:
                                            yield {"type": "partial", "content": chunk}
                                        sentence_buffer = sentence_buffer[break_point+1:].strip()
                                        
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse streaming JSON: {e}")
                                continue
                                
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama streaming API error {response.status}: {error_text}")
                    yield {"type": "error", "content": "I'm having trouble thinking right now. Could you try again?"}
                    
        except Exception as e:
            logger.error(f"Ollama streaming request failed: {e}")
            yield {"type": "error", "content": "I'm experiencing technical difficulties. Please try again."}
    
    async def analyze_image(self, image_data, prompt="What do you see in this image?"):
        """
        Analyze image using vision capabilities
        
        Args:
            image_data: Image as PIL Image or bytes
            prompt (str): Vision prompt
            
        Returns:
            str: Image description
        """
        try:
            # Convert image to base64
            if isinstance(image_data, Image.Image):
                buffer = io.BytesIO()
                image_data.save(buffer, format="JPEG", quality=85)
                image_bytes = buffer.getvalue()
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            else:
                logger.error("Unsupported image data type")
                return "I couldn't process that image."
            
            # Encode to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            session = await self._get_session()
            
            # Vision request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for vision
                    "max_tokens": 100
                }
            }
            
            logger.debug("Sending vision request to Ollama")
            
            async with session.post(f"{self.base_url}/api/chat", 
                                  json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    description = data.get("message", {}).get("content", "")
                    logger.debug(f"Vision response: {description}")
                    return description.strip()
                else:
                    error_text = await response.text()
                    logger.error(f"Vision API error {response.status}: {error_text}")
                    return "I couldn't analyze that image properly."
                    
        except Exception as e:
            logger.error(f"Vision request failed: {e}")
            return "I had trouble processing that image."
    
    async def health_check(self):
        """Check if Ollama is running and model is available"""
        try:
            session = await self._get_session()
            
            # Check if server is running
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return False, "Ollama server not responding"
                
                data = await response.json()
                models = [model["name"] for model in data.get("models", [])]
                
                if self.model_name not in models:
                    return False, f"Model {self.model_name} not found. Available: {models}"
                
                return True, f"Ollama healthy, using {self.model_name}"
                
        except Exception as e:
            return False, f"Ollama health check failed: {e}"
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Ollama session cleaned up")
