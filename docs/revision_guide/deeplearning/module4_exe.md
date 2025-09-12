# Module 4 — Exercises and Reflection

## Exercises

1. Replace freefield1010 with warblrb to train the CNN. Is the accuracy improved or
   worsened?
|   Dataset   |   Accuracy   |
| *********** | ************ |
|**freefield1010**| 0.72 |
|**warblrb10k**| 0.7669 |   
3. Use the TinyChirp dataset to train the CNN. This dataset has fixed train/vali-
   date/test splits. What kind of modifications are needed to properly train your
   CNN?

4. Load a few audio files and plot their waveforms and Mel spectrograms side by
   side. Describe what differences you observe between bird and non-bird clips.

5. Add another convolutional layer to the CNN. What happens the accuracy? Can
   we get better results with a deeper model?

6. Try different audio segment lengths (e.g., 1 second, 2 seconds, 5 seconds) and
   compare model performance.

---

## Reflection

 • Why is stratified splitting important when you have an unbalanced dataset?  
 • How do class weights affect the training of a CNN on unbalanced data?  
 • How does adding a channel dimension for CNN input differ from simply reshaping the array? Why is it necessary for convolutional layers?  
