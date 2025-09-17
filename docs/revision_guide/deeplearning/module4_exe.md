# Module 4 — Exercises and Reflection

## Exercises

1. Replace freefield1010 with warblrb to train the CNN. Is the accuracy improved or worsened?

   | Dataset | Accuracy |
   |---|---|
   | **freefield1010** | 0.7672 |
   | **warblrb10k** | 0.7669 |  

2. Use the TinyChirp dataset to train the CNN. This dataset has fixed train/validate/test splits. What kind of modifications are needed to properly train your CNN?  
- Since the dataset is already categorized into train/validate/test files, bird and non-birds sounds are separated in different files. So there is no necessary to split the dataset. Therefore metadata and data splitting are excluded in modified codes. Modified Codes: [TinyChirp_Model_Training](https://github.com/pohyuwei0111/RaspberryPi_Project/blob/3e6b5b6261a1a513a35440a738861d5686c61684/docs/revision_guide/deeplearning/TinyChirp_BAD_Training.md)
- Final Validation Accuracy: **0.9877**
3. Load a few audio files and plot their waveforms and Mel spectrograms side by side. Describe what differences you observe between bird and non-bird clips.
  
**Bird Clips**  
<img width="1466" height="490" alt="image" src="https://github.com/user-attachments/assets/cfa9f977-809b-465e-9a94-f50e61c21b70" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/f123c53f-55ac-4f51-9eb5-fb8def89b646" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/88d8c9b7-9af7-406c-828b-c2ab846210c7" />
  
**Non-Bird Clips**  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/85b66e57-a096-48f9-abfc-8df31373799b" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/293e22e2-d5a3-41a3-bf47-ec12fb7c01f3" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/dabcaca7-78dd-43f6-a293-e421628e3fb8" />  


4. Add another convolutional layer to the CNN. What happens the accuracy? Can we get better results with a deeper model?

   **Changing kernel size**  
   | Convolutional layer | Accuracy |
   |---|---|
   | **2-layers** | 0.9877 |
   | **3-layers** | 0.9913 |
   | **4-layers** | 0.7188 |
   
   **Keep Kernel size (3,3). Using padding='same'**  
   | Convolutional layer | Accuracy |
   |---|---|
   | **2-layers** | 0.9971 |
   | **3-layers** | 0.9949 |
   | **4-layers** | 0.9957 |

| Experiment                               | Accuracy Trend                              | Reason                                                                                   |
|------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------------------|
| Change kernel size (no padding)          | 2-layers: 0.9877 → 3-layers: 0.9913 → 4-layers: 0.7188 | Without `padding="same"`, feature maps shrink quickly; after several layers, dimensions collapse → network can’t learn effectively → accuracy drops. |
| Keep kernel size (3,3), padding='same'   | 2-layers: 0.9971 → 3-layers: 0.9949 → 4-layers: 0.9957 | `same` padding preserves feature map size, allowing deeper layers to learn without losing spatial information → accuracy stays high and stable. |
| General takeaway                         | Moderate depth improves accuracy, but too many layers can harm | Deeper CNNs capture more complex features, but excessive layers or shrinking feature maps lead to overfitting or invalid dimensions. |


---

## Reflection

 • Why is stratified splitting important when you have an unbalanced dataset?  
 • How do class weights affect the training of a CNN on unbalanced data?  
 • How does adding a channel dimension for CNN input differ from simply reshaping the array? Why is it necessary for convolutional layers?  
 
 | Concept                   | Why it matters on unbalanced data / CNN input | Effect / Benefit |
|----------------------------|-----------------------------------------------|------------------|
| **Stratified Splitting**   | Ensures class proportions are preserved across train/val/test sets | Prevents validation/test sets from missing minority classes; gives reliable evaluation |
| **Class Weights**          | Assigns higher penalty to minority class errors in loss function | Forces model to learn minority classes instead of biasing toward majority; improves balanced performance |
| **Channel Dimension in CNN** | Input must be 4D `(batch, H, W, C)`; CNN kernels expect channels | Preserves spatial patterns; enables convolution over 2D data (e.g., grayscale as 1-channel, RGB as 3-channel) |  

Reshaping flattens → CNN loses ability to detect spatial features. Adding channel dimension preserves 2D structure and lets Conv2D work properly.
