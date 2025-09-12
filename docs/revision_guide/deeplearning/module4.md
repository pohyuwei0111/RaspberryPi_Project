# Bird Activity Detection (BAD) — simplified step-by-step guide (code included)

This is a compact, copy-pasteable Colab / local notebook that follows **Module 4** (training a BAD model on Freefield1010).

---

## Quick notes before you start
- Recommended: run on Google Colab (GPU optional) or a machine with enough RAM.  
- Installs: `librosa`, `tensorflow`, `pandas`, `scikit-learn`, `soundfile` (librosa needs it).  
- The code below reproduces the pipeline in Module 4: download dataset & metadata, extract loudest 3s segment (for bird clips), compute 16-bin Mel spectrograms, train a tiny CNN, then convert to TensorFlow Lite (with optional integer quantization).

---

## 0) Install dependencies (Colab)
```bash
pip install librosa tensorflow pandas scikit-learn soundfile tqdm
```

---

## 1) Imports & configuration
```python
import os
import zipfile
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

DATASET_ZIP_PATH = "/content/freefield1010.zip"
DATASET_EXTRACT_PATH = "/content/freefield1010"
METADATA_CSV_PATH = "/content/freefield1010_metadata.csv"

os.makedirs(os.path.dirname(DATASET_ZIP_PATH), exist_ok=True)
os.makedirs(DATASET_EXTRACT_PATH, exist_ok=True)

DATASET_URL = "https://archive.org/download/ff1010bird/ff1010bird_wav.zip"
METADATA_URL = "https://ndownloader.figshare.com/files/10853303"
```

---

## 2) Download & extract dataset + metadata
```python
def download_dataset():
    if os.path.exists(DATASET_EXTRACT_PATH) and any(fname.endswith('.wav') for _,_,files in os.walk(DATASET_EXTRACT_PATH) for fname in files):
        print("Dataset already extracted, skipping download.")
        return DATASET_EXTRACT_PATH

    if not os.path.exists(DATASET_ZIP_PATH):
        print("Downloading Freefield1010...")
        r = requests.get(DATASET_URL, stream=True)
        total = int(r.headers.get('content-length', 0))
        with open(DATASET_ZIP_PATH, 'wb') as f, tqdm(
                desc="Downloading", total=total, unit='iB', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    print("Extracting zip...")
    with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as z:
        z.extractall(DATASET_EXTRACT_PATH)
    return DATASET_EXTRACT_PATH

def download_metadata():
    if os.path.exists(METADATA_CSV_PATH):
        print("Metadata CSV already present.")
    else:
        print("Downloading metadata CSV...")
        r = requests.get(METADATA_URL)
        r.raise_for_status()
        with open(METADATA_CSV_PATH, 'wb') as f:
            f.write(r.content)
    df = pd.read_csv(METADATA_CSV_PATH)
    label_dict = {str(int(row['itemid'])): int(row['hasbird']) for _, row in df.iterrows()}
    return label_dict
```

---

## 3) Audio helpers: loudest segment & mel extraction
```python
SAMPLE_RATE = 16000
TARGET_CLIP_DURATION = 3.0
SAMPLES_PER_CLIP = int(SAMPLE_RATE * TARGET_CLIP_DURATION)
WINDOW_STEP = 0.1
N_MELS = 16
FFT_SIZE = 512
HOP_LENGTH = 256

def find_loudest_segment(audio, sr, duration=TARGET_CLIP_DURATION, step=WINDOW_STEP):
    samples_per_segment = int(sr * duration)
    step_samples = int(sr * step)
    if len(audio) <= samples_per_segment:
        return np.pad(audio, (0, samples_per_segment - len(audio)), mode='constant')
    max_energy = -np.inf
    best = None
    for start in range(0, len(audio) - samples_per_segment + 1, step_samples):
        seg = audio[start:start + samples_per_segment]
        energy = np.sqrt(np.mean(seg ** 2))
        if energy > max_energy:
            max_energy = energy
            best = seg
    return best

def extract_mel_db(segment):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
```

---

## 4) Process files to features
```python
def process_audio_file(filepath, label):
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        if label == 1:
            segment = find_loudest_segment(audio, sr)
        else:
            if len(audio) >= SAMPLES_PER_CLIP:
                start = np.random.randint(0, len(audio) - SAMPLES_PER_CLIP + 1)
                segment = audio[start:start + SAMPLES_PER_CLIP]
            else:
                segment = np.pad(audio, (0, SAMPLES_PER_CLIP - len(audio)), mode='constant')
        mel_db = extract_mel_db(segment)
        return mel_db, label
    except Exception:
        return None, None

def build_dataset(data_dir, label_dict, max_files=None):
    wav_files = [os.path.join(root, f) for root, _, files in os.walk(data_dir) for f in files if f.endswith('.wav')]
    X_list, y_list = [], []
    for i, fp in enumerate(tqdm(wav_files)):
        if max_files and i >= max_files:
            break
        fid = os.path.splitext(os.path.basename(fp))[0]
        if fid not in label_dict:
            continue
        label = label_dict[fid]
        feat, lab = process_audio_file(fp, label)
        if feat is not None:
            X_list.append(feat)
            y_list.append(lab)
    X = np.array(X_list)[..., np.newaxis].astype(np.float32)
    y = np.array(y_list).astype(np.int32)
    return X, y
```

---

## 5) Prepare dataset
```python
DATA_DIR = download_dataset()
LABEL_DICT = download_metadata()
X, y = build_dataset(DATA_DIR, LABEL_DICT, max_files=None)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class_weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)))
```

---

## 6) Define tiny CNN
```python
input_shape = X_train.shape[1:]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 7) Train model
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    class_weight=class_weights
)
```

---

## 8) Convert to TensorFlow Lite
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("bird_activity_default_quant.tflite", "wb") as f:
    f.write(tflite_model)
```

Optional full integer quantization:
```python
def representative_gen():
    for i in range(min(100, X_train.shape[0])):
        idx = np.random.randint(0, X_train.shape[0])
        sample = X_train[idx:idx+1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
with open("bird_activity_int8.tflite", "wb") as f:
    f.write(tflite_int8)
```
