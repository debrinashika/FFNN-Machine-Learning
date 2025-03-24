import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x))  
    return e / np.sum(e, axis=0, keepdims=True)
