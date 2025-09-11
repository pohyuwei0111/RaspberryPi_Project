# Model Deployment on Raspberry Pi  
## Steps
Train a model -> Quantization -> Convert to tflite -> Deployment
## Train a model  
[Model Training Guide](docs/revision_guide/deeplearning/module1.md)
## Quantization (2 Ways)  
1. Post Training Quantization
```bash
 import tensorflow as tf
 converter = tf.lite.TFLiteConverter.from_keras_model(model)
 converter.optimizations = [tf.lite.Optimize.DEFAULT]
```
2. Quantization-Aware Training
```bash
 import tensorflow as tf
 import tensorflow_model_optimization as tfmot
 model = tf.keras.Sequential([...])
 model.compile(...)
 # Apply quantization-aware training
 quant_aware_model = tfmot.quantization.keras.quantize_model(model)
 quant_aware_model.fit(x_train, y_train, epochs=5)
 # Convert to TFLite
 converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
 converter.optimizations = [tf.lite.Optimize.DEFAULT]
 tflite_model = converter.convert()
```
 ## Conversion to tflite  
 ```bash
 with open("mnist_model_int8.tflite", "wb") as f:
 f.write(tflite_model)
```
## Deployment on Raspberry Pi  
Install ai-edge-litert runtime ( upgraded version of tflite runtime)  
```bash
pip install ai-edge-litert
```
Then run the python scripts to perform inference of MNIST  
[MNIST_Infer](https://github.com/pohyuwei0111/RaspberryPi_Project/blob/7413d62eb2ed2f6c1acdc63b64820696f9a75ba4/src/inference/mnist_infer.py)

# Result of Inference  
## Single Image Inference  
[digit_recog.py](https://github.com/pohyuwei0111/RaspberryPi_Project/blob/d90557c2395fdf49bb24ffeb99d2f73351a49a90/src/inference/digit_recog.py)  

<img width="874" height="123" alt="Screenshot 2025-09-11 102810" src="https://github.com/user-attachments/assets/0261afe2-84cb-4085-a7e2-e8cb3c19b252" />

## 10,000 Image Inference
<img width="817" height="72" alt="image" src="https://github.com/user-attachments/assets/8bd65f4e-1692-4af0-9385-1b52e8c16584" />  

## Confusion Matrix  
<img width="795" height="601" alt="image" src="https://github.com/user-attachments/assets/022a6177-eefc-4651-aeba-961843760a64" />
