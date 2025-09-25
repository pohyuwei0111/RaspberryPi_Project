import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter

# ==== CONFIG ====
MODEL_PATH = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
LABELS_PATH = "labels_en.txt"
SAMPLE_RATE = 48000
DURATION = 3.0
SAMPLES_NEEDED = int(SAMPLE_RATE * DURATION)
CONF_THRESHOLD = 0.5

# ==== LOAD LABELS ====
with open(LABELS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

# ==== LOAD MODEL ====
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(" Starting BirdNET microphone detection (Ctrl+C to stop)...")

try:
    while True:
        # Record 3 seconds
        print("\n[INFO] Recording 3s...")
        recording = sd.rec(SAMPLES_NEEDED, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()

        # Flatten to mono
        y = recording.flatten()

        # Pad/truncate
        if len(y) < SAMPLES_NEEDED:
            y = np.pad(y, (0, SAMPLES_NEEDED - len(y)))
        else:
            y = y[:SAMPLES_NEEDED]

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], y[np.newaxis, :])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Top prediction
        pred_idx = int(np.argmax(output_data))
        confidence = float(output_data[pred_idx])
        pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Unknown({pred_idx})"

        if confidence >= CONF_THRESHOLD:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{ts}.wav"
            sf.write(filename, y, SAMPLE_RATE)
            print(f" Bird detected! {pred_class} ({confidence:.3f})")
            print(f" Saved recording: {filename}")

            # ==== Generate mel spectrogram ====
            S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=(8, 4))
            librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", cmap="magma")
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"{pred_class} ({confidence:.2f})")
            plt.tight_layout()
            plt.show()

            input("Press Enter to continue detection...")

        else:
            print(f" No confident detection (Top={pred_class}, Conf={confidence:.3f})")

except KeyboardInterrupt:
    print("\n Stopped detection.")
