import numpy as np

class Layers:
    def __init__(self, n_inputs: int, n_neurons: int, activ_func, use_bias=1):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activ_func = activ_func
        self.use_bias = use_bias

        self.weight = np.random.randn((n_inputs + use_bias), n_neurons)
