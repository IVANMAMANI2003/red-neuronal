
# 🎓 Red Neuronal para Predicción de Abandono Estudiantil

Este proyecto implementa una red neuronal artificial (RNA) entrenada con un dataset de estudiantes universitarios, con el objetivo de predecir si un estudiante continuará, abandonará o se graduará.

## 📌 Objetivo
Aplicar inteligencia artificial mediante redes neuronales en un contexto real de la carrera de Ingeniería de Sistemas, con una utilidad práctica como sistemas de alerta temprana o módulos predictivos integrados en plataformas educativas.

---

## 🧠 Tecnologías Utilizadas

- Python 3.11
- TensorFlow / Keras
- Pandas, NumPy, Scikit-learn
- Matplotlib (opcional)
- VSCode / Jupyter Notebooks

---

## 📂 Estructura del Proyecto

```
RED-NEURONAL/
├── data/                # Dataset de entrenamiento
│   └── data.csv
├── models/              # Modelos entrenados
│   └── dropout_model.h5
├── src/                 # Código fuente principal
│   ├── preprocessing.py
│   ├── model.py
│   └── main.py
├── predict.py           # Script para probar nuevas predicciones
├── requirements.txt     # Dependencias del proyecto
└── venv/                # Entorno virtual (ignorar en GitHub)
```

---

## ⚙️ Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/red-neuronal-abandono.git
cd red-neuronal-abandono
```

### 2. Crear entorno virtual
```bash
python -m venv venv
```

### 3. Activar entorno virtual
- En Windows (CMD):
```bash
venv\Scripts\activate
```
- En PowerShell:
```bash
.env\Scripts\Activate.ps1
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Entrenar el modelo
```bash
python src/main.py
```

### 6. Hacer predicciones
```bash
python predict.py
```

---

## 🧪 Ejemplo de Predicción

`predict.py` permite ingresar datos simulados y obtener la predicción de abandono:

```bash
Resultado: Se gradúa (probabilidades: [0.01, 0.03, 0.96])
```

---

## 💼 Aplicaciones en Ingeniería de Sistemas

- Desarrollo de sistemas de alerta temprana en universidades
- Dashboards con predicción de retención estudiantil
- Sistemas inteligentes en plataformas educativas
- Módulos de IA integrados en sistemas web

---

## 👨‍🎓 Autor

**[Tu Nombre]**  
Estudiante de Ingeniería de Sistemas - UPeU  
Curso: Análisis Multivariado  
Docente: [Nombre del docente]

---

## 📄 Licencia

Este proyecto es solo para fines educativos.
