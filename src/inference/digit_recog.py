import numpy as np
from ai_edge_litert.interpreter import Interpreter

# 1. Load MNIST dataset
with np.load("/home/poh/Documents/vecad/python/Deep_learning/mnist.npz") as data:
    x_test, y_test = data["x_test"], data["y_test"]

# 2. Normalize and reshape
x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)  # (num, 28, 28, 1)

#  Choose target digit and index
target_digit = 3   # which digit you want (0–9)
sample_index = 0   # pick the first sample of that digit

# 3. Filter test dataset for chosen digit
mask = y_test == target_digit
x_digit = x_test[mask]
y_digit = y_test[mask]

print(f"Found {len(y_digit)} samples of digit {target_digit}")
print(f"Testing sample index {sample_index}")

# 4. Load TFLite model
interpreter = Interpreter(model_path="/home/poh/Documents/vecad/python/Deep_learning/mnist_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 5. Run inference on 1 sample
input_data = x_digit[sample_index].reshape(1, 784).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.argmax(output_data)

print("True label:", y_digit[sample_index])
print("Model prediction:", pred)

