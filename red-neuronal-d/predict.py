import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Cargar modelo
model = load_model('models/dropout_model.h5')

# Cargar datos originales
df = pd.read_csv('data/data.csv', sep=';')

# Igual que en el entrenamiento
drop_cols = ['Course', 'Nacionality', "Daytime/evening attendance\t"]
df.drop(columns=drop_cols, inplace=True)

# Codificar variables categóricas
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separar features y target
X = df.drop('Target', axis=1)
y = df['Target']

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Elegir una fila de prueba (por ejemplo, la número 10)
fila = X_scaled[9].reshape(1, -1)

# Predecir
prediccion = model.predict(fila)
clase = np.argmax(prediccion)
clases = {0: 'Continúa', 1: 'Abandona', 2: 'Se gradúa'}

# Mostrar resultado
print(f"Resultado: {clases[clase]} (probabilidades: {prediccion[0]})")

# Recomendaciones según el resultado
if clase == 0:
    print("Recomendación: Felicitaciones, el estudiante tiene alta probabilidad de continuar. Mantener el apoyo y seguimiento académico.")
elif clase == 1:
    print("Recomendación: El estudiante tiene riesgo de abandono. Se sugiere intervención temprana: tutorías, consejería académica y apoyo emocional.")
elif clase == 2:
    print("Recomendación: El estudiante está próximo a graduarse. Brindar orientación profesional y apoyo en inserción laboral.")

# Matriz de confusión para todo el dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Predecir para todos los estudiantes
y_pred_all = np.argmax(model.predict(X_scaled), axis=1)
cm = confusion_matrix(y, y_pred_all)
print("\nMatriz de Confusión (consola, DATA COMPLETA):")
print(cm)
print("\nFilas = Clase real | Columnas = Clase predicha")
print("Etiquetas: 0=Continúa, 1=Abandona, 2=Se gradúa")

# Visualización gráfica
clases = ["Continúa", "Abandona", "Se gradúa"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - DATA COMPLETA")
plt.show()
