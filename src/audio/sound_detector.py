import pyaudio
import numpy as np

CHUNK = 1024          # number of samples per frame
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100          # samples per second

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        # Read audio data
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Compute amplitude
        amplitude = np.linalg.norm(data) / CHUNK

        # Scale amplitude into bar length
        bar_length = int(amplitude / 10)   # adjust divisor for sensitivity
        bar = "|" * bar_length

        # Print bar + amplitude value
        print(f"{bar} {amplitude:.2f}")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
