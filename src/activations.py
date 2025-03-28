import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(np.clip(x, -100, 100))   
    return e / np.sum(e, axis=1, keepdims=True)

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x / (1 + np.exp(-x))

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))