import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# inputs=[[1,2,3,2.5],
#         [2.0,5.0,-1.0,2.0],
#         [-1.5,2.7,3.3,-0.8]]

# weights=[[0.2, 0.8, -0.5, 1.0],
#         [0.5, -0.91, 0.26, -0.5],
#         [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# weights2=[[0.1, -0.14, 0.5],
#         [-0.5, 0.12, -0.33],
#         [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

#np.random.seed(0)

# X=[[1,2,3,2.5],
#      [2.0,5.0,-1.0,2.0],
#     [-1.5,2.7,3.3,-0.8]]

# X, y = spiral_data(100, 3)

# layer1 = Layer_Dense(2,5)
# # layer2 = Layer_Dense(5,2)

# activation1 = Activation_ReLU()

# layer1.forward(X)

# print(layer1.output)
# activation1.forward(layer1.output)
# print(activation1.output)

#print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

# np.random.seed(0)

# X=[[1,2,3,2.5],
#      [2.0,5.0,-1.0,2.0],
#     [-1.5,2.7,3.3,-0.8]]

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# print(output)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output =  probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CatagoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CatagoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)