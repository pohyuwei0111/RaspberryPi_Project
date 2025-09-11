import numpy as np
from ai_edge_litert.interpreter import Interpreter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load MNIST dataset
with np.load("/home/poh/Documents/vecad/python/Deep_learning/mnist.npz") as data:
    x_test, y_test = data["x_test"], data["y_test"]

# 2. Normalize + flatten to (num_samples, 784)
x_test = x_test.astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784)

# 3. Load TFLite model
interpreter = Interpreter(model_path="/home/poh/Documents/vecad/python/Deep_learning/mnist_model_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. Run inference on all test samples
correct = 0
y_pred = []
total = len(x_test)

for i in range(total):
    input_data = np.expand_dims(x_test[i], axis=0)  # shape (1, 784)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data)
    y_pred.append(pred)

    if pred == y_test[i]:
        correct += 1

y_pred = np.array(y_pred)

accuracy = correct / total * 100
print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("MNIST Confusion Matrix")
plt.show()

