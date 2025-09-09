# Week 1 (2025-09-02 → 2025-09-04)

## Theme / Focus
Headless Pi setup, remote dev, basic audio pipeline, baseline STT.

## Goals (planned)
- [x] Headless Raspberry Pi setup with VNC
- [x] Configure USB mic and test input via Python
- [x] Build amplitude detector
- [x] Integrate Vosk small model for STT
- [x] Continuous recording with auto transcript saving

## Day-by-day log
**Tue (2025-09-02)**  
-Explore alternative way to setup Raspberry Pi

**Wed (2025-09-03)**
- Flashed Raspberry Pi OS (headless), enabled SSH
- Configured Wi-Fi & VNC; verified remote desktop access
- Installed Python deps
- Detected USB mic (`arecord -l`), confirmed capture device
- Wrote quick Python snippet to read audio frames
- Implemented simple amplitude detector

**Thu (2025-09-04)**
- Installed Vosk small model
- Ran first STT on short clips; transcripts saved to txt
- Prototyped continuous recording + rolling STT
- WIP: automatic segmenting and transcript file naming

## Artifacts / Demos
- Code: [src/audio/amplitude_detector.py](../../src/audio/amplitude_detector.py)
- Code (WIP): 
- Notes: 
