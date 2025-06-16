import numpy as np


def step_function(z):
    return 1 if z >= 0 else 0

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return step_function(z)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
            print(f'Epoch {epoch+1}/{epochs} - Weights: {self.weights}')

# Características: [nivel_de_glucosa, presión_sanguínea, edad]
X = np.array([
    [150, 85, 45],
    [80, 70, 25],
    [180, 90, 50],
    [70, 65, 20],
    [160, 88, 55],
    [90, 75, 30],
])


y = np.array([1, 0, 1, 0, 1, 0])


perceptron = Perceptron(input_size=3)
perceptron.train(X, y, epochs=100)

# Probar el perceptrón
test_samples = np.array([
    [100, 80, 35],
    [170, 95, 60],
])

for inputs in test_samples:
    result = perceptron.predict(inputs)
    print(f"Entrada: {inputs} - Salida Predicha: {'Diabetes' if result == 1 else 'No Diabetes'}")
