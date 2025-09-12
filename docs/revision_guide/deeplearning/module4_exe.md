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

5. Add another convolutional layer to the CNN. What happens the accuracy? Can we get better results with a deeper model?

6. Try different audio segment lengths (e.g., 1 second, 2 seconds, 5 seconds) and compare model performance.

---

## Reflection

 • Why is stratified splitting important when you have an unbalanced dataset?  
 • How do class weights affect the training of a CNN on unbalanced data?  
 • How does adding a channel dimension for CNN input differ from simply reshaping the array? Why is it necessary for convolutional layers?  
