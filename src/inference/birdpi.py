import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from ai_edge_litert.interpreter import Interpreter
import os
import librosa
import cv2
import matplotlib.pyplot as plt
import librosa.display
import matplotlib.image as mpimg

# ==== TARGET SPECIES (exact BirdNET labels) ====
TARGET_SPECIES = [
    "Eudynamys scolopaceus_Asian Koel",
    "Oriolus chinensis_Black-naped Oriole",
    "Parus cinereus_Cinereous Tit",
    "Todiramphus chloris_Collared Kingfisher",
    "Aegithina tiphia_Common Iora",
    "Spilornis cheela_Crested Serpent-Eagle",
    "Caprimulgus macrurus_Large-tailed Nightjar",
    "Rhipidura javanica_Malaysian Pied-Fantail",
    "Streptopelia chinensis_Spotted Dove",
    "Geopelia striata_Zebra Dove"
]

# ==== CONFIG ====
BIRDSPECIES_MODEL = "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite"
BAD_MODEL = "bad_model_int8.tflite"   # your Bird Activity Detection model
LABELS_PATH = "labels_en.txt"
SAMPLE_RATE = 48000
DURATION = 3.0
SAMPLES_NEEDED = int(SAMPLE_RATE * DURATION)
CONF_THRESHOLD = -2
BAD_THRESHOLD = 0.1
SAVE_DIR = "detections"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== IMAGE PATHS (map each species to its JPEG) ====
IMAGE_DIR = "bird_png"
SPECIES_IMAGES = {sp: os.path.join(IMAGE_DIR, sp + ".jpeg") for sp in TARGET_SPECIES}

# ==== LOAD BIRDNET LABELS ====
with open(LABELS_PATH, "r") as f:
    ALL_LABELS = [line.strip() for line in f]

# Map target species to indices
TARGET_IDX = {sp: ALL_LABELS.index(sp) for sp in TARGET_SPECIES}

# ==== LOAD MODELS ====
interpreter_bad = Interpreter(model_path=BAD_MODEL)
interpreter_bad.allocate_tensors()
bad_input = interpreter_bad.get_input_details()[0]
bad_output = interpreter_bad.get_output_details()[0]

interpreter_birdnet = Interpreter(model_path=BIRDSPECIES_MODEL)
interpreter_birdnet.allocate_tensors()
bn_input = interpreter_birdnet.get_input_details()[0]
bn_output = interpreter_birdnet.get_output_details()[0]

print(" Starting Bird Activity + BirdNET detection (Ctrl+C to stop)...")

def run_bad(y):
    """Run Bird Activity Detection model"""
    # Preprocess into (1,64,184,1) int8
    S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64, n_fft=1024, hop_length=160, win_length=400)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    S_img = (S_norm * 255).astype(np.uint8)

    # Pad/trim to 184 frames
    if S_img.shape[1] < 184:
        pad = 184 - S_img.shape[1]
        S_img = np.pad(S_img, ((0,0),(0,pad)), mode="constant")
    else:
        S_img = S_img[:, :184]

    input_data = S_img[np.newaxis, ..., np.newaxis].astype(np.int8)

    # Run inference
    interpreter_bad.set_tensor(bad_input['index'], input_data)
    interpreter_bad.invoke()
    output_q = interpreter_bad.get_tensor(bad_output['index'])[0]

    # Dequantize
    scale, zero_point = bad_output['quantization']
    output = (output_q.astype(np.float32) - zero_point) * scale

    # Softmax
    probs = np.exp(output) / np.sum(np.exp(output))
    return probs[1]  # probability of "bird"

try:
    while True:
        # Record 3 seconds
        print("\n[INFO] Recording 3s...")
        recording = sd.rec(SAMPLES_NEEDED, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        y = recording.flatten()

        # Bird Activity Detection (use resampled version at 16kHz)
        y16k = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=16000)
        bird_prob = run_bad(y16k)

        if bird_prob < BAD_THRESHOLD:
            print(f" No bird activity detected (prob={bird_prob:.2f})")
            continue  # skip BirdNET

        print(f" Bird activity detected! (prob={bird_prob:.2f}) → running BirdNET...")

        # Pad/truncate for BirdNET
        if len(y) < SAMPLES_NEEDED:
            y = np.pad(y, (0, SAMPLES_NEEDED - len(y)))
        else:
            y = y[:SAMPLES_NEEDED]

        # Run BirdNET
        interpreter_birdnet.set_tensor(bn_input['index'], y[np.newaxis, :])
        interpreter_birdnet.invoke()
        output_data = interpreter_birdnet.get_tensor(bn_output['index'])[0]

        # Extract scores for target species
        target_scores = {sp: output_data[idx] for sp, idx in TARGET_IDX.items()}
        pred_class, confidence = max(target_scores.items(), key=lambda x: x[1])

        if confidence >= CONF_THRESHOLD:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"detected_{ts}.wav")
            sf.write(filename, y, SAMPLE_RATE)
            print(f" Bird species detected: {pred_class} ({confidence:.3f})")
            print(f" Saved recording: {filename}")
            
         # === Load bird image ===
            img = None
            if pred_class in SPECIES_IMAGES and os.path.exists(SPECIES_IMAGES[pred_class]):
                img = mpimg.imread(SPECIES_IMAGES[pred_class])


            # Plot mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
            S_db = librosa.power_to_db(S, ref=np.max)
            
        # === Plot side by side ===
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Left: bird image
            if img is not None:
                ax[0].imshow(img)
                ax[0].axis("off")
                ax[0].set_title(pred_class, fontsize=12)
            else:
                ax[0].text(0.5, 0.5, "No image available", ha="center", va="center", fontsize=12)
                ax[0].axis("off")

            # Right: spectrogram
            img_spec = librosa.display.specshow(S_db, sr=SAMPLE_RATE, x_axis="time",
                                                y_axis="mel", cmap="magma", ax=ax[1])
            fig.colorbar(img_spec, ax=ax[1], format="%+2.0f dB")
            ax[1].set_title(f"Mel spectrogram\nConf={confidence:.2f}")

            plt.tight_layout()
            plt.show()

            input("Press Enter to continue detection...")
        else:
            print(f" Bird detected but not in top 10 species (best={pred_class}, {confidence:.3f})")

except KeyboardInterrupt:
    print("\n Stopped detection.")

