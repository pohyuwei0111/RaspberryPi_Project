# DSP for Audio
## Exercise  
# Expected Outcomes – Mel Spectrogram Exercises

| Exercise | Parameter Change | Expected Effect on Mel Spectrogram |
|----------|------------------|------------------------------------|
| **FFT Size** | Small FFT (256) | High time resolution → short chirps clear; Low frequency resolution → harmonics blurred. Hard to distinguish frequency parts |
|          | Large FFT (2048) | High frequency resolution → harmonics sharp; Low time resolution → chirps smeared/stretched. |
| **Mel Filters** | Low frequencies | Dense filters → fine detail preserved (fundamental, formants clear). |
|              | High frequencies | Sparse filters → compressed, coarse detail (high-pitch energy looks chunkier). |
| **MFCC Count** | 13 coefficients | Broad spectral envelope only; smooth, less detailed representation. |
|               | 20 coefficients | More fine detail preserved; subtler harmonic changes captured, but higher dimensionality. |
| **Colormaps** | Grayscale | the worst, hard to interpret |
|              | Viridis/Inferno | Perceptually uniform → faint features stand out clearly. |
|              | magma | Easy to spot high energy sound |

**1. Generate a spectrogram of a bird sound with different FFT sizes (e.g., 256, 512,1024). Compare the time–frequency resolution.**  
FFT = 256  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/f4c87ddb-7b79-4b90-bb98-7bf91199f29e" />

FFT = 512  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/968e68c8-d233-4144-ac7c-13493e932ee8" />  

FFT = 1024  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/3b7a76bf-74a3-443c-80ff-87a6c907b4fd" />  

FFT =2048
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/0d3c220d-2f9b-4fb6-bbb4-58c34a69ff80" />  


**2. Compare the resolution of Mel filters in the low-frequency range versus the high-frequency range. Why does the Melscale emphasize low frequencies more?**  
<img width="1038" height="654" alt="image" src="https://github.com/user-attachments/assets/3f90ecc1-a1eb-4efd-bad2-f0a8aaba4c57" />  

**3. Extract MFCCs from the same bird sound using 13 coefficients and then 20 coefficients. Compare the feature sets.**  
13 coefficients  
<img width="907" height="390" alt="image" src="https://github.com/user-attachments/assets/cc7c967d-153a-4e46-8938-e83c712e594a" />  

20 coefficients  
<img width="907" height="390" alt="image" src="https://github.com/user-attachments/assets/9fb83edd-ca2b-46b0-93b7-160235f50a82" />

**4. Try different colormaps (gray_r, magma, viridis) for spectrogram visualization. Which one makes the patterns easiest to see?**  
Viridis  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/c5dd3ea2-9b4b-48c2-ad4e-bf43fde520f4" />

gray_r  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/c6df069f-bee2-4f7e-bbeb-edf7a538df26" />

magma  
<img width="924" height="390" alt="image" src="https://github.com/user-attachments/assets/f6155058-352b-4961-965c-222805e1018e" />
