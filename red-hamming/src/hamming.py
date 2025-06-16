import numpy as np


def hamming_network(W, x, threshold=2.5):
    h = np.dot(W, x)
    f = np.where(h > threshold, 1, 0)
    return h, f
