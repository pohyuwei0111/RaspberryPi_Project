# *Bird Species Classification with CNN (MobileNetV3 + Mel-Spectrograms)*  

---
full codes:  [spec_class.py](https://github.com/pohyuwei0111/RaspberryPi_Project/blob/f9cc8fcbdb7d643f283fefea20ed7c47dd4c7e19/docs/revision_guide/deeplearning/spec_class.py)
- **Pipeline:**
  1. Load audio → convert to Mel-spectrogram → RGB image.  
  2. Preprocess for MobileNetV3.  
  3. Train CNN with transfer learning.  
  4. Evaluate with accuracy, classification report, confusion matrix.  

---

## 🔹 Key Revisions in Module 5

| Step | Revision | Reason |
|------|----------|--------|
| **Dataset Handling** | Switched from loading all spectrograms into RAM → `tf.data.Dataset` with caching. | Prevents RAM crash in Colab. |
| **Splitting** | Manual 80/10/10 split into train/val/test sets. | Keeps a separate test set for fair evaluation. |
| **Spectrogram Function** | Added `wav_to_mel` with resizing to 224×224 and RGB colormap. | Makes input compatible with MobileNetV3 pretrained weights. |
| **Preprocessing** | Applied `preprocess_input(x*255.0)` after Mel-spectrogram conversion. | Ensures input matches ImageNet-trained MobileNetV3 expectations. |
| **Model** | Base: `MobileNetV3Small(include_top=False, weights="imagenet")` → GAP → Dropout → Dense. | Transfer learning for faster convergence. |
| **Caching** | `ds.cache("/content/...")` used for train/val/test pipelines. | First epoch slow, later epochs much faster, avoids slowdown. |
| **Evaluation** | Added sklearn `classification_report`, `confusion_matrix`, and confusion matrix plots. | Provides detailed per-class performance. |
| **Results** | Achieved ~80% test accuracy. | Verified model works properly with preprocessing + caching. |

---

## 🔹 How to Run
1. Upload dataset `.zip` to Colab `/content/Downloads`.  
2. Update:
   ````python
   zip_file_path = "/content/Downloads/seabird3s.zip"
   DATASET_PATH = "/content/seabird3s"

## Test Results  
Evaluating model on test set...  
Test Accuracy: 80.3%

### 📄 Classification Report

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| asikoe2   | 0.817     | 0.927  | 0.869    | 82      |
| colkin1   | 0.741     | 0.800  | 0.769    | 75      |
| comior1   | 0.814     | 0.608  | 0.696    | 79      |
| commyn    | 0.652     | 0.592  | 0.621    | 76      |
| comtai1   | 0.773     | 0.800  | 0.786    | 85      |
| latnig2   | 0.952     | 0.988  | 0.970    | 81      |
| magrob    | 0.686     | 0.886  | 0.773    | 79      |
| olbsun4   | 0.802     | 0.706  | 0.751    | 109     |
| spodov2   | 0.978     | 0.989  | 0.984    | 92      |
| whtkin2   | 0.794     | 0.714  | 0.752    | 70      |  

**Overall Accuracy:** 0.803 (828 samples)  

| Metric        | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Macro Avg     | 0.801     | 0.801  | 0.797    | 828     |
| Weighted Avg  | 0.805     | 0.803  | 0.800    | 828     |

Dataset Statistics:  
-Total samples: 8278  
-Number of classes: 10  
-Test set size: 828 samples (10.0%of total)  
## Confusion Matrix  
<img width="527" height="470" alt="image" src="https://github.com/user-attachments/assets/c10a2fa3-e613-48bb-94a6-662ecae603a4" />

## Sample Predictions  

=== SAMPLE PREDICTIONS ===  
Correct - True: latnig2, Predicted: latnig2 (confidence: 1.00)  
Correct - True: whtkin2, Predicted: whtkin2 (confidence: 0.83)  
Correct - True: asikoe2, Predicted: asikoe2 (confidence: 0.74)  
Correct - True: comior1, Predicted: comior1 (confidence: 0.40)  
Correct - True: comtai1, Predicted: comtai1 (confidence: 0.97)  

## Training and Validation Curve  
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/506bc43a-21bf-44d1-9f70-8dc85d731653" />

