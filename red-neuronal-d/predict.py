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
