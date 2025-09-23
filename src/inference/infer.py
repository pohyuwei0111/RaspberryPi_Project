import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf   # Use TF Lite runtime in Colab
#from ai_edge_litert.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib
from keras.applications import MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input

# ==== CONFIG ====
MODEL_PATH = "bird_species_model_seabird.tflite"  # upload your .tflite here
SAMPLE_RATE = 44100
DURATION = 3.0
N_MELS = 224
FFT_SIZE = 2048
HOP_LENGTH = 512
TARGET_H, TARGET_W = 224, 224   # resize target

CLASS_NAMES = [

        "asikoe2","colkin1","comior1","commyn","comtai1",
    "latnig2","magrob","olbsun4","spodov2","whtkin2"
]

    # "asikoe2","colkin1","comior1","commyn","comtai1",
    # "latnig2","magrob","olbsun4","spodov2","whtkin2"

# 'Asian_Koel', 'Black_naped_Oriole', 'Cinereous_Tit', 'Collared_Kingfisher',
 # 'Common_Iora', 'Crested_serpent_Eagle', 'Large_tailed_Nightjar', 'Pied_Fantail', 'Spotted_Dove', 'Zebra_Dove'

def wav_to_rgb_mel_inference(path, debug=False):
    # Load audio (pad/trim to 3s)
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    samples_needed = int(SAMPLE_RATE * DURATION)
    if len(audio) < samples_needed:
        audio = np.pad(audio, (0, samples_needed - len(audio)))
    else:
        audio = audio[:samples_needed]

    # Mel spectrogram (same params as training)
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=N_MELS, n_fft=FFT_SIZE, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize 0–1
    S_min, S_max = S_db.min(), S_db.max()
    S_norm = (S_db - S_min) / (S_max - S_min + 1e-8)

    # Apply magma colormap → RGB float32 in 0–1
    S_rgb = matplotlib.colormaps['magma'](S_norm)[..., :3]

    # Resize with TensorFlow (same as training)
    S_resized = tf.image.resize(S_rgb, (TARGET_H, TARGET_W)).numpy().astype(np.float32)

    # Preprocess for MobileNetV3: preprocess_input(x*255.0)
    # Equivalent to: (S_resized*255.0)/127.5 - 1.0
    S_pre = preprocess_input((S_resized * 255).astype(np.float32))

    if debug:
        print(f"DEBUG: S_db min/max = {S_min:.3f}/{S_max:.3f}")
        print(f"DEBUG: S_norm min/max = {S_norm.min():.3f}/{S_norm.max():.3f}")
        print(f"DEBUG: S_resized min/max = {S_resized.min():.3f}/{S_resized.max():.3f}")
        print(f"DEBUG: S_pre min/max = {S_pre.min():.3f}/{S_pre.max():.3f}")
        plt.imshow(S_resized,origin="lower", aspect="auto", cmap="magma")
        plt.title("Resized magma RGB (0–1)")
        plt.colorbar()
        plt.show()

    #return S_resized[np.newaxis, ...]
    return S_pre[np.newaxis, ...]   # (1,224,224,3)


def run_inference(wav_path, model_path=MODEL_PATH, top_k=3, debug=False):
    # Preprocess input
    input_data = wav_to_rgb_mel_inference(wav_path, debug=debug)

    # Load model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Print predictions
    print(f"\n🎶 File: {wav_path}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls}: {output_data[i]:.3f}")
    pred_idx = int(np.argmax(output_data))
    print(f"\n✅ Predicted: {CLASS_NAMES[pred_idx]} (confidence {output_data[pred_idx]:.3f})")

    # Top-k
    topk_idx = np.argsort(output_data)[::-1][:top_k]
    print("\nTop-k:")
    for idx in topk_idx:
        print(f" {CLASS_NAMES[idx]}: {output_data[idx]:.3f}")

run_inference("D:/deeplearning/dataset/extract/putra_dataset/Asian_Koel/20200322_060000_HSBU_0_00879_141.wav", debug=True)
run_inference("D:/deeplearning/dataset/extract/putra_dataset/Large_tailed_Nightjar/20200322_000000_HSBU_0_00253_178.wav", debug=True)
run_inference("/deeplearning/dataset/extract/seabird3s/whtkin2/ml61501961_0.wav", debug=True)

#D:/deeplearning/dataset/extract/putra_dataset/Asian_Koel/20200322_060000_HSBU_0_00879_141.wav
#D:/deeplearning/dataset/extract/seabird3s/whtkin2/ml61501961_0.wav
