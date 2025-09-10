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

