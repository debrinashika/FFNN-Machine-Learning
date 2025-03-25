import numpy as np
from layers import Layers 
import activations 
from backpropagation import softmaxOutput, softmaxHidden, tanOutput, tanHidden

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
        self.delta_gradien = [np.zeros((layer.n_inputs+1, layer.n_neurons)) for layer in self.layers]

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
            return np.mean((np.array(target) - np.array(output)) ** 2)
        elif self.loss_func == "binary":
            pass
        elif self.loss_func == "categorical":
            return -np.mean(np.sum(target * np.log(output + 1e-9), axis=1))

    def updateWeight(self):
        for idx, layer in enumerate(self.layers):
            layer.weight += self.delta_gradien[idx][0]

    def updateGradien(self, layer_idx: int, delta: np.ndarray, input: np.ndarray):
        grad = self.learning_rate * np.dot(input, delta)
        self.delta_gradien[layer_idx] = grad


    def addInputOutput(self, input: list[float], output: list[float]):
        self.input.append(input)
        self.target.append(output)

    def addHiddenLayer(self, layer: Layers):
        self.layers.append(layer)

    def feedForward(self):
        self.initWeight()

        for j in range(1000):
            if j == self.epoch:
                return
            
            self.initDeltaGradien()
            error = 0
            
            for i, batch in enumerate(self.input):
                inputs: list[list[float]] = []
                nets: list[list[float]] = []

                batch = np.array(batch)  
                batch_size = batch.shape[0]
                bias = np.ones((batch_size, 1))
                current = np.hstack((bias, batch))  
                inputs.append(current.copy().transpose().tolist())

                for layer in self.layers:
                    net = np.dot(current,layer.weight) 
                    current = layer.activ_func(net)
                    nets.append(net.copy().transpose().tolist())
                    if layer != self.layers[-1]:
                        bias = np.ones((batch_size, 1))
                        current = np.hstack((bias, current))  
                    inputs.append(current.copy().transpose().tolist())

                error += self.calcLoss(self.target[i],current) 
                print(f"Epoch {j+1}, Loss: {error}")
                # print(inputs, "\n")
                # print(nets)
                self.backPropagation(inputs, nets, self.target[i])
                self.updateWeight()
                self.initDeltaGradien()

    def backPropagation(self, inputs, netsLayer, target):
        i = len(self.layers) - 1
        delta1: np.ndarray = None
        
        while i >= 0:
            nets = np.array(netsLayer[i]).T 
   
            if i == len(self.layers) - 1:  # Output layer
                if self.layers[i].activ_func == activations.softmax:
                    delta1 = softmaxOutput(inputs[i+1], target)
                elif self.layers[i].activ_func == activations.tanh:
                    delta1 = tanOutput(inputs[i+1], nets, target)  
            else:  # Hidden layer
                # print("LAYERR",self.layers[i + 1].weight)
                if self.layers[i].activ_func == activations.softmax:
                    delta2 = softmaxHidden(self.layers[i + 1].weight[1:], inputs[i+1][1:], delta1)  
                elif self.layers[i].activ_func == activations.tanh:
                    delta2 = tanHidden(self.layers[i + 1].weight[1:], inputs[i+1][1:], delta1)

                self.updateGradien(i + 1, delta1, self.layers[i + 1].weight)
                delta1 = delta2

            i -= 1



                
            
