import numpy as np

class RMSNorm:
    def __init__(self, d: int, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.gamma = np.ones(d) 

    def forward(self, x: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return (x / rms) * self.gamma
