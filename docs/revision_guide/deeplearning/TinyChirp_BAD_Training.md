# TinyChirp Bird Activity Detection (BAD) Training Script

This script trains a simple CNN on the **TinyChirp** dataset (downloaded from Kaggle), 
extracts Mel spectrogram features, and converts the trained model to TensorFlow Lite (TFLite).

```python
import os
import numpy as np
import tensorflow as tf
import librosa
import kagglehub
from sklearn.utils import class_weight

#------------------------------------------------------
# Download TinyChirp dataset from Kaggle
#------------------------------------------------------
path = kagglehub.dataset_download("zhaolanhuang/tinychirp")
print("Path to dataset files:", path)

#------------------------------------------------------
# Audio preprocessing utilities
#------------------------------------------------------
def find_loudest_segment(audio, sr, duration=3.0, step=0.1):
    samples_per_segment = int(sr * duration)
    step_samples = int(sr * step)
    max_energy = -np.inf
    best_segment = None

    if len(audio) < samples_per_segment:
        return np.pad(audio, (0, samples_per_segment - len(audio)), mode='constant')

    for start in range(0, len(audio) - samples_per_segment + 1, step_samples):
        segment = audio[start:start + samples_per_segment]
        energy = np.sqrt(np.mean(segment**2))
        if energy > max_energy:
            max_energy = energy
            best_segment = segment

    if best_segment is None:
        best_segment = audio[:samples_per_segment]
        best_segment = np.pad(best_segment, (0, max(0, samples_per_segment - len(best_segment))))
    return best_segment

# Audio parameters
SAMPLE_RATE = 16000
TARGET_CLIP_DURATION = 3.0
SAMPLES_PER_CLIP = int(SAMPLE_RATE * TARGET_CLIP_DURATION)
WINDOW_STEP = 0.1
N_MELS = 16
FFT_SIZE = 512
HOP_LENGTH = 256

def process_audio_file(filepath, label):
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        if label == 1:
            segment = find_loudest_segment(audio, sr, TARGET_CLIP_DURATION, WINDOW_STEP)
        else:
            if len(audio) >= SAMPLES_PER_CLIP:
                start = np.random.randint(0, len(audio) - SAMPLES_PER_CLIP + 1)
                segment = audio[start:start + SAMPLES_PER_CLIP]
            else:
                segment = np.pad(audio, (0, SAMPLES_PER_CLIP - len(audio)), mode='constant')

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

#------------------------------------------------------
# Load train, validation, and test splits
#------------------------------------------------------
TRAIN_PATH = os.path.join(path, "training")
VAL_PATH   = os.path.join(path, "validation")
TEST_PATH  = os.path.join(path, "testing")

X_train, y_train = build_dataset_from_folders(TRAIN_PATH)
X_val, y_val     = build_dataset_from_folders(VAL_PATH)
X_test, y_test   = build_dataset_from_folders(TEST_PATH)

#------------------------------------------------------
# Handle class imbalance
#------------------------------------------------------
USE_CLASS_WEIGHTS = True
class_weights = None
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
        USE_CLASS_WEIGHTS = False
else:
    print("Class weights disabled.")

#------------------------------------------------------
# Define CNN model
#------------------------------------------------------
print("Defining model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax') # 2 classes: target / non-target
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

#------------------------------------------------------
# Train model
#------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32
training_msg = f"Training model for {EPOCHS} epochs"
if USE_CLASS_WEIGHTS and class_weights:
    training_msg += " with class weights..."
else:
    training_msg += " without class weights..."
print(training_msg)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights if USE_CLASS_WEIGHTS and class_weights else None
)

#------------------------------------------------------
# Convert to TensorFlow Lite
#------------------------------------------------------
print("Converting model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

suffix = "_with_weights" if USE_CLASS_WEIGHTS else "_no_weights"
tflite_filename = f"bird_activity_simplified_TinyChirp{suffix}.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved as '{tflite_filename}'")

# Final evaluation
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
```
