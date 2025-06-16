import numpy as np

# Datos de entrada: [asistencia, promedio]
X = np.array([[0.1, 0.3],
              [0.4, 0.5],
              [0.7, 0.8],
              [0.9, 0.6]])

# Salida esperada: aprobado (1) o no aprobado (0)
y = np.array([0, 0, 1, 1])

# Inicializar pesos y bias
pesos = np.random.rand(2)
bias = np.random.rand(1)
tasa_aprendizaje = 0.1
epocas = 10

# Función de activación
def step_function(x):
    return 1 if x >= 0.5 else 0

# Entrenamiento
for epoch in range(epocas):
    print(f"Época {epoch + 1}")
    for i in range(len(X)):
        entrada = X[i]
        salida_esperada = y[i]
        z = np.dot(entrada, pesos) + bias
        prediccion = step_function(z)

        error = salida_esperada - prediccion

        # Actualización de pesos y bias
        pesos += tasa_aprendizaje * error * entrada
        bias += tasa_aprendizaje * error

        print(f"Entrada: {entrada}, Predicción: {prediccion}, Esperado: {salida_esperada}, Error: {error}")
    print(f"Pesos: {pesos}, Bias: {bias}\n")

# Prueba
entrada_prueba = np.array([0.8, 0.9])
z = np.dot(entrada_prueba, pesos) + bias
resultado = step_function(z)
print(f"Entrada de prueba: {entrada_prueba}, Resultado: {resultado}")
