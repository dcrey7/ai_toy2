/**
 * AI Voice Assistant - v2.2 Frontend Script
 * Implements streaming audio and client-side audio playback fixes.
 */

class VoiceAssistant {
    constructor() {
        this.socket = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.audioContext = null; // Master audio context
        
        this.audioState = 'idle'; // idle|ai_speaking|user_listening|processing
        
        // Voice-to-voice latency timing
        this.voiceTimingMetrics = {
            userSpeechEnd: null,
            aiSpeechStart: null,
            currentResponseId: null
        };
        
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.controlsContainer = document.getElementById('controlsContainer');
        this.chatInputContainer = document.getElementById('chatInputContainer');
        
        this.init();
    }
    
    async init() {
        this.setupUI();
        this.connectWebSocket();
        this.initCamera();
    }
    
    
    setupUI() {
        const controlsHTML = `
            <div class="voice-controls-section">
                <div class="status-indicator">
                    🎥 Live<br>
                    <span id="connectionStatus">🔴 Connecting...</span>
                </div>
                <div class="system-status">
                    <div class="status-item">
                        STT:<br>
                        <span class="status-value" id="sttStatus">🟢 Ready<br><small>Kyutai STT-1B</small></span>
                    </div>
                    <div class="status-item">
                        LLM:<br>
                        <span class="status-value" id="llmStatus">🟢 Ready<br><small>Gemma3:1B</small></span>
                    </div>
                    <div class="status-item">
                        TTS:<br>
                        <span class="status-value" id="ttsStatus">🟢 Ready<br><small>Kokoro TTS</small></span>
                    </div>
                    <div class="status-item">
                        GPU:<br>
                        <span class="status-value" id="gpuStatus">⚪ CPU Only<br><small>NVIDIA RTX 3050 6GB</small></span>
                    </div>
                    <div class="status-item">
                        VAD:<br>
                        <span class="status-value" id="vadSystemStatus">🟢 Ready<br><small>Smart Turn v2</small></span>
                    </div>
                </div>
                <div class="mode-toggle">
                    <button id="textModeBtn" class="mode-button active">💬 Text Chat</button>
                    <button id="voiceModeBtn" class="mode-button">🎤 Voice Chat</button>
                </div>
                <button id="voiceToggle" class="voice-button start" style="display: none;">🎤 Start Conversation</button>
                <div id="vadStatus" class="vad-status"></div>
            </div>
        `;
        this.controlsContainer.innerHTML = controlsHTML;
        
        this.textModeBtn = document.getElementById('textModeBtn');
        this.voiceModeBtn = document.getElementById('voiceModeBtn');
        this.voiceToggle = document.getElementById('voiceToggle');
        this.vadStatus = document.getElementById('vadStatus');
        this.connectionStatus = document.getElementById('connectionStatus');
        
        this.textModeBtn.addEventListener('click', () => this.switchToTextMode());
        this.voiceModeBtn.addEventListener('click', () => this.switchToVoiceMode());
        this.voiceToggle.addEventListener('click', () => this.toggleVoice());
        this.sendButton.addEventListener('click', () => this.sendTextMessage());
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendTextMessage(); }
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        this.socket = new WebSocket(wsUrl);
        this.socket.onopen = () => { this.updateConnectionStatus('🟢 Connected'); window.logger.success('🔌 WebSocket connected'); };
        this.socket.onmessage = (event) => this.handleWebSocketMessage(event.data);
        this.socket.onclose = () => { this.updateConnectionStatus('🔴 Disconnected'); setTimeout(() => this.connectWebSocket(), 3000); };
        this.socket.onerror = () => { this.updateConnectionStatus('🔴 Error'); };
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            switch (message.type) {
                case 'transcript': this.addMessage('user', message.text); break;
                case 'response': 
                    // Only handle non-streaming responses (fallback compatibility)
                    if (!this.currentStreamingMessage) {
                        this.addMessage('assistant', message.text); 
                    }
                    break;
                case 'streaming_start': this.startStreamingMessage('assistant'); break;
                case 'streaming_chunk': this.appendToStreamingMessage(message.text); break;
                case 'streaming_complete': this.completeStreamingMessage(); break;
                case 'status': this.updateSystemStatus(message.component, message.status); break;
                case 'vad_status': this.updateVadStatus(message.status); break;
                case 'recording_control': this.handleRecordingControl(message); break;
                case 'tts_audio':
                    this.handleTTSAudio(message);
                    break;
                case 'log': 
                    // Check for response ID in log messages to track timing
                    if (message.message && message.message.includes('[Response:')) {
                        const responseIdMatch = message.message.match(/\[Response: (.+?)\]/);
                        if (responseIdMatch && message.message.includes('Turn complete detected')) {
                            this.voiceTimingMetrics.userSpeechEnd = performance.now();
                            this.voiceTimingMetrics.currentResponseId = responseIdMatch[1];
                        }
                    }
                    window.logger.log(message.level, message.message); 
                    break;
                case 'live_metrics': this.updateLiveMetrics(message); break;
                case 'performance_metrics': this.updatePerformanceMetrics(message); break;
                case 'metrics_response': this.handleMetricsResponse(message); break;
                default: window.logger.debug(JSON.stringify(message));
            }
        } catch (error) {
            window.logger.error(`WebSocket parse error: ${error.message}`);
        }
    }
    
    
    
    handleTTSAudio(message) {
        // Mark AI speech start for voice-to-voice latency
        if (this.voiceTimingMetrics.currentResponseId && !this.voiceTimingMetrics.aiSpeechStart) {
            this.voiceTimingMetrics.aiSpeechStart = performance.now();
            
            // Calculate and send voice-to-voice latency
            if (this.voiceTimingMetrics.userSpeechEnd) {
                const voiceToVoiceLatency = (this.voiceTimingMetrics.aiSpeechStart - this.voiceTimingMetrics.userSpeechEnd) / 1000;
                
                // Send client-side timing to backend
                this.sendVoiceTimingMetrics({
                    responseId: this.voiceTimingMetrics.currentResponseId,
                    voiceToVoiceLatency: voiceToVoiceLatency,
                    timestamp: Date.now()
                });
                
                window.logger.success(`🎯 Voice-to-Voice Latency: ${voiceToVoiceLatency.toFixed(3)}s [${this.voiceTimingMetrics.currentResponseId}]`);
            }
        }
        
        this.playTTSAudio(message.audio, message.sample_rate);
    }
    
    switchToTextMode() {
        this.textModeBtn.classList.add('active');
        this.voiceModeBtn.classList.remove('active');
        this.voiceToggle.style.display = 'none';
        this.vadStatus.style.display = 'none';
        this.chatInputContainer.style.display = 'flex';
        if (this.isRecording) this.stopVoice();
    }
    
    switchToVoiceMode() {
        this.voiceModeBtn.classList.add('active');
        this.textModeBtn.classList.remove('active');
        this.voiceToggle.style.display = 'block';
        this.vadStatus.style.display = 'block';
        this.chatInputContainer.style.display = 'none';
    }
    
    async toggleVoice() {
        if (!this.isRecording) await this.startVoice();
        else this.stopVoice();
    }
    
    async startVoice() {
        try {
            // Use WebSocket audio implementation
            await this.startWebSocketVoice();
            
        } catch (error) {
            console.error('Voice setup error:', error);
            
            // Provide more specific error messages
            let errorMessage = '';
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Microphone access denied. Please allow microphone permissions and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'No microphone found. Please connect a microphone and try again.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Microphone is already in use by another application.';
            } else if (error.message.includes('createScriptProcessorNode')) {
                errorMessage = 'Browser audio processing not supported. Please try a modern browser like Chrome or Firefox.';
            } else {
                errorMessage = `Voice setup failed: ${error.message}`;
            }
            
            window.logger.error(`❌ ${errorMessage}`);
            alert(errorMessage);
            
            // Reset voice mode on error
            this.voiceModeBtn.classList.remove('active');
            this.textModeBtn.classList.add('active');
            this.switchToTextMode();
        }
    }
    
    
    async startWebSocketVoice() {
        try {
            // 1. Create audio context optimized for 16kHz (Smart Turn v2 requirement)
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 16000  // Smart Turn v2 expects 16kHz
                });
            }
            await this.audioContext.resume();

            // 2. Get microphone stream optimized for voice AI
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    echoCancellation: true,      // Essential for voice assistants
                    noiseSuppression: true,      // Remove background noise
                    autoGainControl: true,       // Normalize volume
                    channelCount: 1,             // Mono for efficiency
                    sampleRate: 16000,           // Smart Turn v2 optimal rate
                    sampleSize: 16               // 16-bit depth
                } 
            });

            // 3. Set up Web Audio API for real-time PCM streaming using modern AudioWorklet
            const source = this.audioContext.createMediaStreamSource(this.audioStream);
            
            // Create a ScriptProcessor fallback for browsers that don't support AudioWorklet
            // Use 4096 samples (~256ms at 16kHz) for good balance of latency vs stability
            try {
                // Try modern AudioWorklet first (preferred method)
                await this.audioContext.audioWorklet.addModule('/static/pcm-processor.js');
                this.audioProcessor = new AudioWorkletNode(this.audioContext, 'pcm-processor');
                
                // Handle PCM data from AudioWorklet
                this.audioProcessor.port.onmessage = (event) => {
                    if (this.socket.readyState === WebSocket.OPEN && this.audioState === 'user_listening') {
                        const pcmData = event.data;
                        const base64Data = btoa(String.fromCharCode.apply(null, new Uint8Array(pcmData.buffer)));
                        
                        this.socket.send(JSON.stringify({ 
                            type: 'pcm_chunk', 
                            data: base64Data,
                            sampleRate: this.audioContext.sampleRate,
                            channels: 1,
                            samples: pcmData.length
                        }));
                    }
                };
                
                window.logger.success('🎵 Using modern AudioWorklet for PCM streaming');
                
            } catch (error) {
                // Fallback to ScriptProcessorNode for older browsers
                window.logger.info('📻 Falling back to ScriptProcessorNode');
                
                this.audioProcessor = this.audioContext.createScriptProcessorNode(4096, 1, 1);
                
                // 4. Real-time PCM streaming to Smart Turn v2 VAD
                this.audioProcessor.onaudioprocess = (audioProcessingEvent) => {
                    if (this.socket.readyState === WebSocket.OPEN && this.audioState === 'user_listening') {
                        const inputBuffer = audioProcessingEvent.inputBuffer;
                        const inputData = inputBuffer.getChannelData(0); // Mono channel
                        
                        // Convert Float32 to Int16 for efficient transmission
                        const int16Array = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {
                            // Clamp and convert to 16-bit signed integer
                            int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                        }
                        
                        // Send raw PCM chunk to Smart Turn v2 VAD
                        const base64Data = btoa(String.fromCharCode.apply(null, new Uint8Array(int16Array.buffer)));
                        
                        this.socket.send(JSON.stringify({ 
                            type: 'pcm_chunk', 
                            data: base64Data,
                            sampleRate: this.audioContext.sampleRate,
                            channels: 1,
                            samples: inputData.length
                        }));
                    }
                };
            }
            
            // 5. Connect audio processing graph
            source.connect(this.audioProcessor);
            this.audioProcessor.connect(this.audioContext.destination); // Connect to destination to prevent GC
            
            // 6. Update UI and enable hands-free voice mode
            this.isRecording = true;
            this.audioState = 'user_listening';
            this.voiceToggle.textContent = '⏹️ Stop Conversation';
            this.voiceToggle.className = 'voice-button stop';
            this.socket.send(JSON.stringify({ type: 'toggle_voice_mode', enabled: true }));
            
            window.logger.success('🎤 Real-time PCM streaming started - hands-free conversation enabled!');
            
        } catch (error) {
            throw error;
        }
    }
    
    stopVoice() {
        // Clean up Web Audio API components
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        // Stop audio stream
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        
        this.isRecording = false;
        this.voiceToggle.textContent = '🎤 Start Conversation';
        this.voiceToggle.className = 'voice-button start';
        this.socket.send(JSON.stringify({ type: 'toggle_voice_mode', enabled: false }));
        this.updateVadStatus('inactive');
        
        window.logger.info('🔇 Real-time PCM streaming stopped');
    }

    handleRecordingControl(message) {
        // Smart Turn v2 handles turn detection automatically
        // This is kept for compatibility but not used in hands-free mode
        const action = message.action;
        window.logger.debug(`Recording control: ${action}`);
    }
    
    sendTextMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        this.addMessage('user', message);
        this.chatInput.value = '';
        this.socket.send(JSON.stringify({ type: 'text_message', text: message }));
    }
    
    playTTSAudio(base64Audio, sampleRate) {
        try {
            if (!this.audioContext) this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioContext.resume();
            const binaryString = atob(base64Audio);
            const len = binaryString.length;
            const bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
            const int16Data = new Int16Array(bytes.buffer);
            const floatData = new Float32Array(int16Data.length);
            for (let i = 0; i < int16Data.length; i++) floatData[i] = int16Data[i] / 32768;
            const audioBuffer = this.audioContext.createBuffer(1, floatData.length, sampleRate);
            audioBuffer.copyToChannel(floatData, 0);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start(0);
        } catch (error) {
            window.logger.error(`TTS playback failed: ${error.message}`);
        }
    }
    
    updateConnectionStatus(status, className) {
        this.connectionStatus.textContent = status;
    }
    
    updateSystemStatus(component, status, modelName = '') {
        const statusElement = document.getElementById(`${component}Status`);
        if (!statusElement) {
            console.warn(`Status element not found: ${component}Status`);
            return;
        }
        
        let statusIcon = '🟢';
        let statusText = 'Ready';
        let fullModelName = modelName;
        
        // Set status icon based on status
        switch (status.toLowerCase()) {
            case 'ready':
                statusIcon = '🟢';
                statusText = 'Ready';
                break;
            case 'working':
            case 'processing':
                statusIcon = '🟡';
                statusText = 'Working';
                break;
            case 'error':
                statusIcon = '🔴';
                statusText = 'Error';
                break;
            default:
                // Handle complex status messages like "🟡 Transcribing..."
                if (status.includes('🟡') || status.includes('Working') || status.includes('processing')) {
                    statusIcon = '🟡';
                    statusText = 'Working';
                } else if (status.includes('🟢') || status.includes('Ready')) {
                    statusIcon = '🟢';
                    statusText = 'Ready';
                } else if (status.includes('🔴') || status.includes('Error')) {
                    statusIcon = '🔴';
                    statusText = 'Error';
                } else {
                    statusIcon = '⚪';
                    statusText = status;
                }
        }
        
        // Set model names for different components
        switch (component) {
            case 'stt':
                fullModelName = fullModelName || 'Kyutai STT-1B';
                break;
            case 'llm':
                fullModelName = fullModelName || 'Gemma3:1B';
                break;
            case 'tts':
                fullModelName = fullModelName || 'Kokoro TTS';
                break;
            case 'gpu':
                fullModelName = fullModelName || 'NVIDIA RTX 3050 6GB';
                break;
            case 'vadSystem':
                fullModelName = fullModelName || 'Smart Turn v2';
                break;
        }
        
        statusElement.innerHTML = `${statusIcon} ${statusText}<br><small>${fullModelName}</small>`;
        console.log(`Updated ${component} status: ${statusIcon} ${statusText} - ${fullModelName}`);
    }

    updateVadStatus(status) {
        if (!this.vadStatus) return;
        this.vadStatus.className = `vad-status ${status}`;
        
        switch (status) {
            case 'speaking':
                this.vadStatus.textContent = '🎤 You are speaking...';
                break;
            case 'processing':
                this.vadStatus.textContent = '🧠 Processing your message...';
                break;
            case 'ai_speaking':
                this.vadStatus.textContent = '🤖 AI is speaking...';
                break;
            case 'listening':
                this.vadStatus.textContent = '👂 Listening for your voice...';
                break;
            case 'inactive':
            default:
                this.vadStatus.textContent = '⚪ Voice mode inactive';
                break;
        }
    }
    
    addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        const textNode = document.createTextNode(content);
        const contentDiv = document.createElement('div');
        contentDiv.appendChild(textNode);
        const timeDiv = document.createElement('div');
        timeDiv.className = 'timestamp';
        timeDiv.textContent = new Date().toLocaleTimeString();
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        this.chatMessages.appendChild(messageDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    startStreamingMessage(type) {
        // Create a new streaming message container
        this.currentStreamingMessage = document.createElement('div');
        this.currentStreamingMessage.className = `message ${type} streaming`;
        
        this.streamingContentDiv = document.createElement('div');
        this.streamingContentDiv.className = 'streaming-content';
        
        // Add a typing indicator
        const typingIndicator = document.createElement('span');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.textContent = '▋';
        this.streamingContentDiv.appendChild(typingIndicator);
        
        this.currentStreamingMessage.appendChild(this.streamingContentDiv);
        this.chatMessages.appendChild(this.currentStreamingMessage);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        // Store reference to typing indicator for removal later
        this.typingIndicator = typingIndicator;
    }
    
    appendToStreamingMessage(text) {
        if (!this.currentStreamingMessage || !this.streamingContentDiv) {
            window.logger.warning('No active streaming message to append to');
            return;
        }
        
        // Remove typing indicator temporarily
        if (this.typingIndicator && this.typingIndicator.parentNode) {
            this.typingIndicator.parentNode.removeChild(this.typingIndicator);
        }
        
        // Append new text
        const textSpan = document.createElement('span');
        textSpan.textContent = text + ' ';
        this.streamingContentDiv.appendChild(textSpan);
        
        // Add typing indicator back
        this.streamingContentDiv.appendChild(this.typingIndicator);
        
        // Smooth scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    completeStreamingMessage() {
        if (!this.currentStreamingMessage || !this.streamingContentDiv) {
            window.logger.warning('No active streaming message to complete');
            return;
        }
        
        // Remove typing indicator
        if (this.typingIndicator && this.typingIndicator.parentNode) {
            this.typingIndicator.parentNode.removeChild(this.typingIndicator);
        }
        
        // Remove streaming class to finalize styling
        this.currentStreamingMessage.classList.remove('streaming');
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.className = 'timestamp';
        timeDiv.textContent = new Date().toLocaleTimeString();
        this.currentStreamingMessage.appendChild(timeDiv);
        
        // Clear references
        this.currentStreamingMessage = null;
        this.streamingContentDiv = null;
        this.typingIndicator = null;
        
        // Final scroll
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    sendVoiceTimingMetrics(timingData) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'voice_timing_metrics',
                ...timingData
            }));
        }
        
        // Reset timing data for next response
        this.voiceTimingMetrics = {
            userSpeechEnd: null,
            aiSpeechStart: null,
            currentResponseId: null
        };
    }

    updateLiveMetrics(message) {
        // Update system resource metrics in UI
        try {
            const metricsHtml = `
                <div class="metrics-row">
                    <span>🖥️ CPU: ${message.cpu_percent?.toFixed(1)}%</span>
                    <span>💾 RAM: ${message.ram_percent?.toFixed(1)}%</span>
                    <span>🎮 GPU: ${message.gpu_utilization?.toFixed(1)}%</span>
                    <span>📊 GPU Mem: ${message.gpu_memory_percent?.toFixed(1)}%</span>
                </div>
            `;
            
            // Find or create metrics display in logs
            let metricsDisplay = document.getElementById('liveMetricsDisplay');
            if (!metricsDisplay) {
                metricsDisplay = document.createElement('div');
                metricsDisplay.id = 'liveMetricsDisplay';
                metricsDisplay.className = 'live-metrics-display';
                metricsDisplay.innerHTML = '<div class="metrics-header">📊 Live System Metrics</div>';
                
                // Insert at top of logs container
                const logsContainer = document.getElementById('logsContainer');
                if (logsContainer) {
                    logsContainer.insertBefore(metricsDisplay, logsContainer.firstChild);
                }
            }
            
            // Update content
            const existingRow = metricsDisplay.querySelector('.metrics-row');
            if (existingRow) {
                existingRow.outerHTML = metricsHtml;
            } else {
                metricsDisplay.innerHTML += metricsHtml;
            }
            
        } catch (error) {
            console.error('Error updating live metrics:', error);
        }
    }
    
    updatePerformanceMetrics(message) {
        // Display performance metrics (voice latency, tokens/sec, etc.)
        try {
            const data = message.data || {};
            window.logger.info(`📈 Performance: Voice Latency: ${(data.avg_voice_latency * 1000)?.toFixed(0)}ms, Tokens/sec: ${data.avg_tokens_per_second?.toFixed(1)}`);
        } catch (error) {
            console.error('Error displaying performance metrics:', error);
        }
    }
    
    handleMetricsResponse(message) {
        // Handle metrics API responses
        console.log('Metrics response:', message.request_type, message.data);
        
        if (message.request_type === 'performance_summary') {
            this.updatePerformanceMetrics(message);
        }
    }

    async initCamera() {
        try {
            const video = document.getElementById('videoFeed');
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
            video.srcObject = stream;
            document.getElementById('videoStatus').textContent = '🎥 Live';
        } catch (err) {
            document.getElementById('videoStatus').textContent = '❌ No Camera';
        }
    }
}

function switchTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
    
    // Show selected tab and content
    const selectedTab = document.getElementById(`${tabName}Tab`);
    const selectedContent = document.getElementById(`${tabName}Content`);
    
    if (selectedTab && selectedContent) {
        selectedTab.classList.add('active');
        selectedContent.classList.remove('hidden');
        
        // Show/hide chat input based on tab
        const chatInputContainer = document.getElementById('chatInputContainer');
        if (tabName === 'chat') {
            chatInputContainer.style.display = 'flex';
        } else {
            chatInputContainer.style.display = 'none';
        }
    }
}

class Logger {
    constructor() {
        this.logsContainer = document.getElementById('logsContainer');
        this.maxLogs = 500;
        this.log('info', 'Logger initialized');
    }
    log(level, message) {
        if (!this.logsContainer) return;
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${level}`;
        logEntry.innerHTML = `<span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span> `;
        logEntry.appendChild(document.createTextNode(message));
        this.logsContainer.appendChild(logEntry);
        if (this.logsContainer.children.length > this.maxLogs) {
            this.logsContainer.removeChild(this.logsContainer.firstChild);
        }
        this.logsContainer.scrollTop = this.logsContainer.scrollHeight;
    }
    info(message) { this.log('info', message); }
    warning(message) { this.log('warning', message); }
    error(message) { this.log('error', message); }
    success(message) { this.log('success', message); }
    debug(message) { this.log('debug', message); }
}

document.addEventListener('DOMContentLoaded', () => {
    window.logger = new Logger();
    window.voiceAssistant = new VoiceAssistant();
});
