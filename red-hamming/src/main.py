import csv

import numpy as np

from hamming import hamming_network

proyectos = [
    "Desarrollo Web",
    "Aplicación Móvil",
    "Inteligencia Artificial",
    "Redes y Seguridad",
    "Sistemas Empresariales"
]

W = np.array([
    [1, 1, 1, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 1]
])

with open('./data/entradas.csv') as f:
    reader = csv.reader(f)
    next(reader)  # saltar cabecera
    for i, row in enumerate(reader):
        x = np.array(list(map(int, row[1:9])))
        h, f = hamming_network(W, x)
        if np.sum(f) == 0:
            resultado = "Ninguno supera el umbral"
        else:
            resultado = proyectos[np.argmax(h)]
        print(f"Entrada {i+1}: h = {h.tolist()} → {resultado}")
