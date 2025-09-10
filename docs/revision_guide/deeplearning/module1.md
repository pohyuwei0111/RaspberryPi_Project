# Model Training
```bash
import tensorflow as tf
 print(tf.__version__)
```
```bash
 import tensorflow as tf
 from tensorflow import keras
 import numpy as np
 (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```
```bash
 x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
 x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0
```
```bash
 model = keras.Sequential([
 keras.layers.Dense(128, activation="relu", input_shape=(784,)),
 keras.layers.Dense(10, activation="softmax")
 ])
```
```bash
model.compile(optimizer="adam",
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"])
```
```bash
 model.fit(x_train, y_train, epochs=5, batch_size=32)
```
Result :   
Epoch 1/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 2ms/step - accuracy: 0.8767 - loss: 0.4350
Epoch 2/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 14s 5ms/step - accuracy: 0.9617 - loss: 0.1258
Epoch 3/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 3ms/step - accuracy: 0.9767 - loss: 0.0765
Epoch 4/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.9848 - loss: 0.0532
Epoch 5/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9875 - loss: 0.0415
```bash
test_loss, test_acc = model.evaluate(x_test, y_test)
 print("Test accuracy:", test_acc)
```
Result :  
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9723 - loss: 0.0899  
Test accuracy: 0.9768999814987183
```bash
converter = tf.lite.TFLiteConverter.from_keras_model(model)
 tflite_model = converter.convert()
```
```bash
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  with open("mnist_model_int8.tflite", "wb") as f:
  f.write(tflite_model)
```
# Exercises
## 1. Change the number of neurons in the hidden layer  
**Increase the neurons**
```bash
 model = keras.Sequential([
 keras.layers.Dense(256, activation="relu", input_shape=(784,)),
 keras.layers.Dense(10, activation="softmax")
 ])
```
Result :  
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.9727 - loss: 0.0905
Test accuracy: 0.9757999777793884  
**Decrease the neurons**  
```bash
 model = keras.Sequential([
 keras.layers.Dense(64, activation="relu", input_shape=(784,)),
 keras.layers.Dense(10, activation="softmax")
 ])
```
Result :  
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.9655 - loss: 0.1095  
Test accuracy: 0.9713000059127808  
## 2. Change the activation function from ReLU to sigmoid  
```bash
 model = keras.Sequential([
 keras.layers.Dense(128, activation="sigmoid", input_shape=(784,)),
 keras.layers.Dense(10, activation="softmax")
 ])
```
Result :  
Epoch 1/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 2ms/step - accuracy: 0.8494 - loss: 0.6338  
Epoch 2/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 2ms/step - accuracy: 0.9410 - loss: 0.2060  
Epoch 3/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9585 - loss: 0.1423  
Epoch 4/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.9666 - loss: 0.1167  
Epoch 5/5  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9742 - loss: 0.0906  
Test accuracy: 0.9715999960899353  
## 3. Train for 1, 5, and 10 epochs. Compare results  
**1 epoch**
```bash
 model.fit(x_train, y_train, epochs=1, batch_size=32)
```
Result :  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 2ms/step - accuracy: 0.8733 - loss: 0.4423  
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9501 - loss: 0.1590
Test accuracy: 0.9569000005722046  
**10 epoch**
```bash
 model.fit(x_train, y_train, epochs=10, batch_size=32)
```
Result :  
Epoch 1/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.8780 - loss: 0.4266  
Epoch 2/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.9651 - loss: 0.1187  
Epoch 3/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9762 - loss: 0.0797  
Epoch 4/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9825 - loss: 0.0592  
Epoch 5/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 6s 2ms/step - accuracy: 0.9872 - loss: 0.0427  
Epoch 6/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9900 - loss: 0.0334  
Epoch 7/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.9922 - loss: 0.0260  
Epoch 8/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9939 - loss: 0.0209  
Epoch 9/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.9951 - loss: 0.0165  
Epoch 10/10  
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 5s 2ms/step - accuracy: 0.9952 - loss: 0.0154  
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - accuracy: 0.9720 - loss: 0.1044  
Test accuracy: 0.9772999882698059  
# MNIST Model Training Comparison

| Experiment | Training Time (per epoch) | Final Train Accuracy | Final Test Accuracy |
|------------|----------------------------|----------------------|---------------------|
| **Baseline (128 neurons, ReLU, 5 epochs)** | ~5s/epoch | 98.75% | **97.69%** |
| **More neurons (256, ReLU, 5 epochs)** | ~5s/epoch | ~98.7% | 97.58% |
| **Fewer neurons (64, ReLU, 5 epochs)** | ~5s/epoch | ~96.5% | 97.13% |
| **Change activation (128 neurons, Sigmoid, 5 epochs)** | ~5s/epoch | 97.42% | 97.16% |
| **Short training (1 epoch, 128 neurons, ReLU)** | ~6s total | 87.33% | 95.69% |
| **Long training (10 epochs, 128 neurons, ReLU)** | ~5s/epoch | 99.52% | **97.73%** |



