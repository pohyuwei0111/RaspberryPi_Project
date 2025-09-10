# Model Deployment on Raspberry Pi  
## Steps
Train a model -> Quantization -> Convert to tflite
## Train a model  
## Quantization  
```bash
 import tensorflow as tf
 converter = tf.lite.TFLiteConverter.from_keras_model(model)
 converter.optimizations = [tf.lite.Optimize.DEFAULT]
```
 ## Conversion to tflite  
 ```bash
 tflite_model = converter.convert()
 with open("mnist_model_int8.tflite", "wb") as f:
 f.write(tflite_model)
```
