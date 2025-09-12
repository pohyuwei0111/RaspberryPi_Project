# Module 4 — Exercises and Reflection

## Exercises

1. Replace freefield1010 with warblrb to train the CNN. Is the accuracy improved or worsened?

   | Dataset | Accuracy |
   |---|---|
   | **freefield1010** | 0.7672 |
   | **warblrb10k** | 0.7669 |  

3. Use the TinyChirp dataset to train the CNN. This dataset has fixed train/validate/test splits. What kind of modifications are needed to properly train your CNN?  
- Since the dataset is already categorized into train/validate/test files, bird and non-birds sounds are separated in different files. So there is no necessary to split the dataset. Therefore metadata and data splitting are excluded in modified codes. Modified Codes: [TinyChirp_Model_Training](https://github.com/pohyuwei0111/RaspberryPi_Project/blob/3e6b5b6261a1a513a35440a738861d5686c61684/docs/revision_guide/deeplearning/TinyChirp_BAD_Training.md)
- Final Validation Accuracy: **0.9877**
4. Load a few audio files and plot their waveforms and Mel spectrograms side by side. Describe what differences you observe between bird and non-bird clips.
  
**Bird Clips**  
<img width="1466" height="490" alt="image" src="https://github.com/user-attachments/assets/cfa9f977-809b-465e-9a94-f50e61c21b70" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/f123c53f-55ac-4f51-9eb5-fb8def89b646" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/88d8c9b7-9af7-406c-828b-c2ab846210c7" />
  
**Non-Bird Clips**  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/85b66e57-a096-48f9-abfc-8df31373799b" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/293e22e2-d5a3-41a3-bf47-ec12fb7c01f3" />  
<img width="1465" height="490" alt="image" src="https://github.com/user-attachments/assets/dabcaca7-78dd-43f6-a293-e421628e3fb8" />  


5. Add another convolutional layer to the CNN. What happens the accuracy? Can we get better results with a deeper model?


---

## Reflection

 • Why is stratified splitting important when you have an unbalanced dataset?  
 • How do class weights affect the training of a CNN on unbalanced data?  
 • How does adding a channel dimension for CNN input differ from simply reshaping the array? Why is it necessary for convolutional layers?  
