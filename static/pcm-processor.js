/**
 * PCM Audio Processor for Smart Turn v2 VAD
 * Modern AudioWorklet implementation for real-time audio processing
 */

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input && input[0]) {
            const inputData = input[0]; // Mono channel
            
            for (let i = 0; i < inputData.length; i++) {
                this.buffer[this.bufferIndex] = inputData[i];
                this.bufferIndex++;
                
                // When buffer is full, send PCM data to main thread
                if (this.bufferIndex >= this.bufferSize) {
                    // Convert Float32 to Int16 for efficient transmission
                    const int16Array = new Int16Array(this.bufferSize);
                    for (let j = 0; j < this.bufferSize; j++) {
                        // Clamp and convert to 16-bit signed integer
                        int16Array[j] = Math.max(-32768, Math.min(32767, this.buffer[j] * 32768));
                    }
                    
                    // Send to main thread
                    this.port.postMessage(int16Array);
                    
                    // Reset buffer
                    this.bufferIndex = 0;
                }
            }
        }
        
        // Continue processing
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);