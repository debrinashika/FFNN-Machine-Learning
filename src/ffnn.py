import numpy as np
from layers import Layers  

class FFNN:
    def __init__(self, batch_size: int, learning_rate: float, epoch: int, verbose: int, loss_func, weight_init, seed=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
        self.loss_func = loss_func
        self.weight_init = weight_init  
        self.seed = seed  

        self.input: list[list[float]] = []
        self.target: list[list[float]] = []
        self.layers: list[Layers] = []
        self.delta_gradien: list[np.ndarray] = []

    def initDeltaGradien(self):
        self.delta_gradien = [np.zeros((layer.n_inputs, layer.n_neurons)) for layer in self.layers]

    def initWeight(self, lower_bound=-0.5, upper_bound=0.5, mean=0.0, variance=1.0):
        if self.seed is not None:
            np.random.seed(self.seed)  

        for layer in self.layers:
            if self.weight_init == "zero":
                layer.weight = np.zeros((layer.n_inputs + 1, layer.n_neurons))

            elif self.weight_init == "uniform":
                layer.weight = np.random.uniform(low=lower_bound, high=upper_bound, size=(layer.n_inputs + 1, layer.n_neurons))

            elif self.weight_init == "normal":
                layer.weight = np.random.normal(loc=mean, scale=np.sqrt(variance), size=(layer.n_inputs + 1, layer.n_neurons))

    def calcLoss(self, output: list[float], target: list[float]):
        if self.loss_func == "mse":
            return np.mean((np.array(target) - np.array(output)) ** 2) / 2.0
        elif self.loss_func == "binary":
            pass
        elif self.loss_func == "categorical":
            pass

    def updateWeight(self):
        for idx, layer in enumerate(self.layers):
            layer.weight += self.delta_gradien[idx]

    def updateGradien(self, layer_idx: int, delta: np.ndarray, input: np.ndarray):
        grad = self.learning_rate * np.outer(input, delta)
        self.delta_gradien[layer_idx] += grad

    def addInputOutput(self, input: list[float], output: list[float]):
        self.input.append(input)
        self.target.append(output)

    def addHiddenLayer(self, layer: Layers):
        self.layers.append(layer)

    def feedForward(self):
        self.initWeight()

        for i in range(1000):
            if i == self.epoch:
                return
            
            # self.initDeltaGradien()
            error = 0
            
            for i, batch in enumerate(self.input):
                batch = np.array(batch)  
                batch_size = batch.shape[0]
                bias = np.ones((batch_size, 1))
                current = np.hstack((bias, batch))  

                for layer in self.layers:
                    print("Sebelum aktivasi:", current)
                    net = np.dot(current,layer.weight) 
                    current = layer.activ_func(net)
                    print("Setelah aktivasi:", current)  

                    if layer != self.layers[-1]:
                        bias = np.ones((batch_size, 1))
                        current = np.hstack((bias, current))  

                error += self.calcLoss(self.target[i],current[0])
            
            print(f"Epoch {i+1}, Loss: {error}")

            
           
