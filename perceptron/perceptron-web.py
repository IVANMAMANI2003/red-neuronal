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

    def train(self, X, y, epochs=50):
        for epoch in range(epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
            print(f"Época {epoch + 1}: Pesos = {self.weights}")

# Datos de entrenamiento
X = np.array([
    [5, 0, 0],    # Poco tiempo, sin scroll, sin clic = abandono
    [20, 1, 1],   # Más tiempo, scroll y clic = se quedó
    [10, 0, 1],   # Algo de tiempo, hizo clic = se quedó
    [2, 0, 0],    # Muy poco tiempo = abandono
    [25, 1, 0],   # Buen tiempo y scroll = se quedó
    [3, 0, 0]     # Muy poco tiempo = abandono
])

y = np.array([0, 1, 1, 0, 1, 0])  # Etiquetas (0 = se fue, 1 = se quedó)

# Entrenar perceptrón
perceptron = Perceptron(input_size=3)
perceptron.train(X, y, epochs=20)

# Pruebas
test_samples = np.array([
    [15, 1, 1],   # Usuario muy activo
    [2, 0, 0],    # Usuario pasivo
])

print("\nResultados de prueba:")
for inputs in test_samples:
    result = perceptron.predict(inputs)
    print(f"Entrada: {inputs} → {'Se quedó' if result == 1 else 'Abandonó'}")
