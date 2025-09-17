# TinyChirp Bird Activity Detection (BAD) Training Script

This script trains a simple CNN on the **TinyChirp** dataset (downloaded from Kaggle), 
extracts Mel spectrogram features, and converts the trained model to TensorFlow Lite (TFLite).

import os
 import numpy as np
 import tensorflow as tf
 import librosa
 import zipfile
 import requests
 import pandas as pd
 import kagglehub
 from sklearn.model_selection import train_test_split
 from sklearn.utils import class_weight
 #------------------------------------------------------
 # Paths for dataset and metadata
 #------------------------------------------------------
 #DATASET_ZIP_PATH = "/contents/tiny.zip"
 #DATASET_EXTRACT_PATH = "contents/freefield1010"
 # Create parent directories if they do not exist
 #os.makedirs(os.path.dirname(DATASET_ZIP_PATH), exist_ok=True)
 #os.makedirs(os.path.dirname(DATASET_EXTRACT_PATH), exist_ok=True)
 #------------------------------------------------------
 # URLs for downloading dataset and metadata
 #------------------------------------------------------
 # freefield1010 dataset https://archive.org/download/ff1010bird/ff1010bird_wav.zip
 # freefield1010 metadata https://ndownloader.figshare.com/files/10853303
 # warblrb10k dataset https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip
 # warblrb10k metadata https://ndownloader.figshare.com/files/10853306

 #DATASET_URL = "https://archive.org/download/ff1010bird/ff1010bird_wav.zip"
 #METADATA_URL = "https://ndownloader.figshare.com/files/10853303"

# Download TinyChirp dataset
path = kagglehub.dataset_download("zhaolanhuang/tinychirp")
print("Path to dataset files:", path)

def find_loudest_segment(audio, sr, duration=3.0, step=0.1):
  samples_per_segment = int(sr * duration)
  step_samples = int(sr * step)
  max_energy = -np.inf
  best_segment = None
  # Check if audio is long enough for at least one segment
  if len(audio) < samples_per_segment:
    # Pad if too short
    return np.pad(audio, (0, samples_per_segment - len(audio)), mode='constant')
  for start in range(0, len(audio) - samples_per_segment + 1, step_samples):
    segment = audio[start:start + samples_per_segment]
    # Calculate Root Mean Square (RMS) energy
    energy = np.sqrt(np.mean(segment**2))
    if energy > max_energy:
      max_energy = energy
      best_segment = segment
  # In case the loop doesn’t run (e.g., very short audio relative to step),
  # return the start
  if best_segment is None:
    best_segment = audio[:samples_per_segment]
    best_segment = np.pad(best_segment, (0, max(0, samples_per_segment - len(best_segment))))
  return best_segment

# Audio processing parameters
SAMPLE_RATE = 16000
TARGET_CLIP_DURATION = 3.0
SAMPLES_PER_CLIP = int(SAMPLE_RATE * TARGET_CLIP_DURATION)
WINDOW_STEP = 0.1 # 100ms step for find_loudest_segment
N_MELS = 16
FFT_SIZE = 512
HOP_LENGTH = 256
def process_audio_file(filepath, label):
  """Loads an audio file, extracts a segment based on label, and computes its
  Mel spectrogram."""
  try:
    # 1. Load audio
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    # 2. Extract segment
    # Targeting the loudest segments ensures the model trains on the most
    # informative audio portions,
    # since bird sounds may not span the entire clip.
    if label == 1:
      # For bird calls, find the loudest segment
      segment = find_loudest_segment(audio, sr, TARGET_CLIP_DURATION,
                                     WINDOW_STEP)
    else:
      # For non-bird, take a random segment to introduce variability
      if len(audio) >= SAMPLES_PER_CLIP:
        start = np.random.randint(0, len(audio)- SAMPLES_PER_CLIP + 1)
        segment = audio[start : start + SAMPLES_PER_CLIP]
      else:
        # If audio is shorter than desired clip, pad it
        segment = np.pad(audio, (0, SAMPLES_PER_CLIP-len(audio)),
                         mode='constant')
    # 3. Extract Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db, label
  except Exception as e:
    print(f"Error processing {filepath}: {e}")
    return None, None

def build_dataset_from_folders(base_dir):
    X_list, y_list = [], []
    for label_name, label in [("non_target", 0), ("target", 1)]:
        folder = os.path.join(base_dir, label_name)
        for fname in os.listdir(folder):
            if fname.endswith(".wav"):
                filepath = os.path.join(folder, fname)
                feat, lab = process_audio_file(filepath, label)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(lab)
    X = np.array(X_list)[..., np.newaxis].astype(np.float32)
    y = np.array(y_list).astype(np.int32)
    return X, y

TRAIN_PATH = os.path.join(path, "training")
VAL_PATH   = os.path.join(path, "validation")
TEST_PATH  = os.path.join(path, "testing")

X_train, y_train = build_dataset_from_folders(TRAIN_PATH)
X_val, y_val     = build_dataset_from_folders(VAL_PATH)
X_test, y_test   = build_dataset_from_folders(TEST_PATH)

USE_CLASS_WEIGHTS = True # Set to False to disable class weights
class_weights = None # Default to None
if USE_CLASS_WEIGHTS:
  print("Computing class weights for imbalanced dataset...")
  try:
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights enabled: {class_weights}")
  except ValueError as e:
    print(f"Warning: Could not compute class weights: {e}. Training without class weights.")
    class_weights = None
    USE_CLASS_WEIGHTS = False # Disable flag if computation fails
else:
  print("Class weights disabled.")

print("Defining model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu',padding='same', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax') # 2 classes: bird/no-bird
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Training configuration
EPOCHS = 10
BATCH_SIZE = 32
training_msg = f"Training model for {EPOCHS} epochs"
if USE_CLASS_WEIGHTS and class_weights:
  training_msg += " with class weights..."
else:
  training_msg += " without class weights..."
print(training_msg)
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights if USE_CLASS_WEIGHTS and class_weights else None
)

print("Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Apply default optimizations (includes post-training quantization)
tflite_model = converter.convert()
suffix = "_with_weights" if USE_CLASS_WEIGHTS else "_no_weights"
tflite_filename = f"bird_activity_simplified_TinyChirp{suffix}.tflite"
with open(tflite_filename, "wb") as f:
  f.write(tflite_model)
print(f"TFLite model saved as '{tflite_filename}'")
# Optional: Print final validation accuracy
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Accuracy: {val_accuracy:.4f}")

"""
------
Waveform & Mel Spetrogram Plotting

------

"""

import random
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Example: path to TinyChirp test dataset
TEST_PATH = os.path.join(path, "testing")

# Pick a bird and a non-bird file
bird_file = os.path.join(TEST_PATH, "target", os.listdir(os.path.join(TEST_PATH, "target"))[0])
bird_file1 = os.path.join(TEST_PATH, "target", os.listdir(os.path.join(TEST_PATH, "target"))[1])
bird_file2 = os.path.join(TEST_PATH, "target", os.listdir(os.path.join(TEST_PATH, "target"))[2])
nonbird_file = os.path.join(TEST_PATH, "non_target", os.listdir(os.path.join(TEST_PATH, "non_target"))[0])
nonbird_file1 = os.path.join(TEST_PATH, "non_target", os.listdir(os.path.join(TEST_PATH, "non_target"))[1])
nonbird_file2 = os.path.join(TEST_PATH, "non_target", os.listdir(os.path.join(TEST_PATH, "non_target"))[2])

print("Bird clip:", bird_file)
print("Non-bird clip:", nonbird_file)

def plot_waveform_and_mel(file_path, sr=16000):
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=256, n_mels=16
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Waveform
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")

    # Mel spectrogram
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax[1]
    )
    ax[1].set_title("Mel Spectrogram (dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")

    plt.tight_layout()
    plt.show()

print("Bird Clip")
plot_waveform_and_mel(bird_file)
plot_waveform_and_mel(bird_file1)
plot_waveform_and_mel(bird_file2)
print("Non-Bird Clip")
plot_waveform_and_mel(nonbird_file)
plot_waveform_and_mel(nonbird_file1)
plot_waveform_and_mel(nonbird_file2)

# Final evaluation
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
```
