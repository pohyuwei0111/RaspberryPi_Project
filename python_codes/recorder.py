import os
import sys
import pyaudio
import wave
import json
import datetime
from vosk import Model, KaldiRecognizer

# ==== SETTINGS ====
SAMPLE_RATE = 16000
CHUNK = 4000
SAVE_DIR = "recordings"

# ==== PREPARE FOLDER ====
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== LOAD MODEL ====
if not os.path.exists("model"):
    print("Please download the model and unpack as 'model' in the current folder.")
    sys.exit(1)

model = Model("model")
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# ==== START MICROPHONE ====
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=8000)
stream.start_stream()

# ==== FILE NAMES ====
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
wav_path = os.path.join(SAVE_DIR, f"{timestamp}.wav")
txt_path = os.path.join(SAVE_DIR, f"{timestamp}.txt")

frames = []
transcript = []

print(" Recording started... (Press Ctrl+C to stop)")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                transcript.append(text)
                print("You said:", text)

except KeyboardInterrupt:
    print("\n Recording stopped by user.")

    # Save audio file
    wf = wave.open(wav_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Save transcript
    with open(txt_path, "w") as f:
        f.write(" ".join(transcript) + "\n")

    print(f"✅ Saved audio: {wav_path}")
    print(f"✅ Saved transcript: {txt_path}")

    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

