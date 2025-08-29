# STT Engine Upgrade to Kyutai - 2025-08-29 at 15:00

This update addresses final bug fixes and completes the major requested upgrade of the Speech-to-Text and Voice Activity Detection engine.

## 1. Core Component Upgrade: STT and VAD

The entire STT/VAD pipeline has been replaced, as requested.

-   **Removed:** The previous `Whisper` and `Silero VAD` based system.
-   **Integrated:** The new **Kyutai STT** model (`kyutai/stt-1b-en_fr-trfs`).
-   **Benefit:** This provides a much higher quality, streaming-native transcription with a built-in semantic Voice Activity Detection, which should resolve the issues with poor transcription quality. A new service (`kyutai_service.py`) was created to handle this model.

## 2. Final Bug Fixes

-   **Duplicate Chat Replies:** A bug in the frontend logic that caused the AI's text response to appear twice has been fixed.
-   **Kokoro TTS Quality:** A text sanitization step was added before synthesizing speech. This removes markdown characters that were causing the high-quality Kokoro voice to fail and fall back to the lower-quality `espeak` voice. All TTS should now use the correct Kokoro engine.

The application is now feature-complete based on all requests. The architecture has been refactored for streaming, all major bugs have been addressed, and the core STT/VAD engine has been upgraded.
