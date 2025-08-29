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
                <div class="status-indicator"><div class="status-dot" id="connectionDot"></div><span id="connectionStatus">Connecting...</span></div>
                <div class="system-status">
                    <div class="status-item"><span class="status-label">STT:</span><span class="status-value" id="sttStatus">⚪ Loading</span></div>
                    <div class="status-item"><span class="status-label">LLM:</span><span class="status-value" id="llmStatus">⚪ Loading</span></div>
                    <div class="status-item"><span class="status-label">TTS:</span><span class="status-value" id="ttsStatus">⚪ Loading</span></div>
                    <div class="status-item"><span class="status-label">GPU:</span><span class="status-value" id="gpuStatus">⚪ Loading</span></div>
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
        this.connectionDot = document.getElementById('connectionDot');
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
        this.socket.onopen = () => { this.updateConnectionStatus('🟢 Connected', 'ready'); window.logger.success('🔌 WebSocket connected'); };
        this.socket.onmessage = (event) => this.handleWebSocketMessage(event.data);
        this.socket.onclose = () => { this.updateConnectionStatus('🔴 Disconnected', 'inactive'); setTimeout(() => this.connectWebSocket(), 3000); };
        this.socket.onerror = () => { this.updateConnectionStatus('🔴 Error', 'inactive'); };
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            switch (message.type) {
                case 'transcript': this.addMessage('user', message.text); break;
                case 'response': this.addMessage('assistant', message.text); break;
                case 'status': this.updateSystemStatus(message.component, message.status); break;
                case 'vad_status': this.updateVadStatus(message.status); break;
                case 'tts_audio':
                    this.playTTSAudio(message.audio, message.sample_rate);
                    // The text is already added from the 'response' message type.
                    // if (message.text) this.addMessage('assistant', message.text);
                    break;
                case 'log': window.logger.log(message.level, message.message); break;
                default: window.logger.debug(JSON.stringify(message));
            }
        } catch (error) {
            window.logger.error(`WebSocket parse error: ${error.message}`);
        }
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
            // 1. Create/resume master audio context on user gesture
            if (!this.audioContext) this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioContext.resume();

            this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 } });
            this.mediaRecorder = new MediaRecorder(this.audioStream, { mimeType: 'audio/webm;codecs=opus' });
            
            // 2. When data is available, send it immediately to the backend
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.socket.readyState === WebSocket.OPEN) {
                    const reader = new FileReader();
                    reader.onload = () => {
                        const base64Data = reader.result.split(',')[1];
                        this.socket.send(JSON.stringify({ type: 'audio_chunk', data: base64Data }));
                    };
                    reader.readAsDataURL(event.data);
                }
            };
            
            this.mediaRecorder.start(250); // Send audio chunks every 250ms
            this.isRecording = true;
            this.voiceToggle.textContent = '⏹️ Stop Conversation';
            this.voiceToggle.className = 'voice-button stop';
            this.socket.send(JSON.stringify({ type: 'toggle_voice_mode', enabled: true }));
        } catch (error) {
            alert('Failed to start voice recording. Please check microphone permissions.');
        }
    }
    
    stopVoice() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') this.mediaRecorder.stop();
        if (this.audioStream) this.audioStream.getTracks().forEach(track => track.stop());
        this.isRecording = false;
        this.voiceToggle.textContent = '🎤 Start Conversation';
        this.voiceToggle.className = 'voice-button start';
        this.socket.send(JSON.stringify({ type: 'toggle_voice_mode', enabled: false }));
        this.updateVadStatus('inactive');
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
        this.connectionDot.className = `status-dot ${className}`;
    }
    
    updateSystemStatus(component, status) {
        const statusElement = document.getElementById(`${component}Status`);
        if (statusElement) statusElement.textContent = status;
    }

    updateVadStatus(status) {
        if (!this.vadStatus) return;
        this.vadStatus.className = `vad-status ${status}`;
        if (status === 'speaking') this.vadStatus.textContent = '🎤 You are speaking...';
        else if (status === 'processing') this.vadStatus.textContent = '🧠 Thinking...';
        else this.vadStatus.textContent = '⚪ Waiting for you to speak';
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
