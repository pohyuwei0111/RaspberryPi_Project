import os
import sys
import pyaudio
import json
from vosk import Model, KaldiRecognizer

# Load model
if not os.path.exists("model"):
    print("Please download the model and unpack as 'model' in the current folder.")
    sys.exit(1)

model = Model("model")
recognizer = KaldiRecognizer(model, 16000)

# Start microphone stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000)
stream.start_stream()

print("🎤 Speak into the microphone...")

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print("You said:", result.get("text", ""))
