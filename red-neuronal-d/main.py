from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from src.model import build_model
from src.preprocessing import load_data, preprocess

# Cargar y preparar datos
df = load_data('data/data.csv')
X, y = preprocess(df)
y_cat = to_categorical(y, num_classes=3)

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Crear y entrenar modelo
model = build_model(X.shape[1])
model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión: {acc*100:.2f}%")

# Matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Predecir clases para el conjunto de prueba
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusión (consola):")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Continúa", "Abandona", "Se gradúa"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# Guardar modelo
model.save('models/dropout_model.h5')
