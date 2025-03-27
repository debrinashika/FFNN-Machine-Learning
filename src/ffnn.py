import numpy as np
from layers import Layers 
import activations 
from backpropagation import outputLayer,hiddenLayer
import networkx as nx
import matplotlib.pyplot as plt

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
        
        if self.weight_init == "custom":
            bxh = np.array([[0.35, 0.35]]) 
            wxh = np.array([[0.15, 0.25], 
                            [0.2, 0.3]])
            self.layers[0].weight = np.vstack((bxh, wxh))  
            bhy = np.array([[0.6, 0.6]])  
            why = np.array([[0.4, 0.5], 
                            [0.45, 0.55]])
            self.layers[1].weight = np.vstack((bhy, why)) 
        else:
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
            output = np.array(output, dtype=np.float64)
            target = np.array(target, dtype=np.float64)
            return -np.mean(target * np.log(output + 1e-9) + (1 - target) * np.log(1 - output))
        elif self.loss_func == "categorical":
            return -np.mean(np.sum(target * np.log(output + 1e-9), axis=1))

    def updateWeight(self):
        for idx, layer in enumerate(self.layers):
            print(f"Layer {idx} before: {layer.weight}")
            layer.weight -= self.delta_gradien[idx]
            print(f"Layer {idx} after: {layer.weight}")

        

    def updateGradien(self, layer_idx: int, delta: np.ndarray, input: np.ndarray):
        grad = self.learning_rate * (np.array(input) @ np.array(delta).T)
        self.delta_gradien[layer_idx] = grad


    def addInputOutput(self, input: list[float], output: list[float]):
        self.input.append(input)
        self.target.append(output)

    def addHiddenLayer(self, layer: Layers):
        self.layers.append(layer)

    def plot_weight_distribution(self):
        for i, layer in enumerate(self.layers):
            weights = layer.weight.flatten()  # Ubah bobot jadi 1D array
            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=30, alpha=0.7, color='b', edgecolor='black')
            plt.title(f'Distribusi Bobot - Layer {i+1}')
            plt.xlabel('Nilai Bobot')
            plt.ylabel('Frekuensi')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

    def plot_gradient_distribution(self):
        for i, grad in enumerate(self.delta_gradien):
            grad = grad.flatten()  # Ubah gradien jadi 1D array
            plt.figure(figsize=(6, 4))
            plt.hist(grad, bins=30, alpha=0.7, color='r', edgecolor='black')
            plt.title(f'Distribusi Gradien - Layer {i+1}')
            plt.xlabel('Nilai Gradien')
            plt.ylabel('Frekuensi')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

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
                # print(f"current : {current}")
                inputs.append(current.copy().transpose().tolist())
                # print(f"input: {inputs}")

                for layer in self.layers:
                    # print(f"weight: {layer.weight}")
                    net = np.dot(current,layer.weight) 
                    current = layer.activ_func(net)
                    nets.append(net.copy().transpose().tolist())
                    # print(f"nets: {nets}")
                    if layer != self.layers[-1]:
                        bias = np.ones((batch_size, 1))
                        current = np.hstack((bias, current))  
                    inputs.append(current.copy().transpose().tolist())
                    print(f"input: {inputs}")

                error += self.calcLoss(current, self.target[i]) 
       
                self.backPropagation(inputs, nets, self.target[i])
                self.updateWeight()
                self.initDeltaGradien()
                print(f"Epoch {j+1}, Loss: {error}")
                print("\n")

    def backPropagation(self, inputs, netsLayer, target):
        print(f"Total layers : {len(self.layers)}")
        i = len(self.layers) - 1
        delta1: np.ndarray = None
        
        while i >= -1:
            nets = np.array(netsLayer[i]).T 

            if i == len(self.layers) - 1:  # Output layer
                if self.layers[i].activ_func == activations.softmax:
                    delta1 = outputLayer(inputs[i+1], nets, target,"softmax",self.loss_func)
                elif self.layers[i].activ_func == activations.tanh:
                    delta1 = outputLayer(inputs[i+1], nets, target,"tanh",self.loss_func)
                elif self.layers[i].activ_func == activations.linear:
                    delta1 = outputLayer(inputs[i+1], nets, target,"linear",self.loss_func)
                
            else:  # Hidden layer
                # print("LAYERR",self.layers[i + 1].weight)
                if self.layers[i].activ_func == activations.softmax:
                    delta2 = hiddenLayer(self.layers[i + 1].weight[1:], inputs[i+1][1:], delta1, "softmax")  
                elif self.layers[i].activ_func == activations.tanh:
                    delta2 = hiddenLayer(self.layers[i + 1].weight[1:], inputs[i+1][1:], delta1, "tanh")  
                elif self.layers[i].activ_func == activations.linear:
                    delta2 = hiddenLayer(self.layers[i + 1].weight[1:], inputs[i+1][1:], delta1, "linear")  

                self.updateGradien(i + 1, delta1,inputs[i+1])

                print(f"Current input: {inputs[i+1][1:]}")
                print(f"Current delta1: {delta1}")
                print(f"Current All gradien: {self.delta_gradien}")
                print(f"Current gradien: {self.delta_gradien[i + 1]}")
                delta1 = delta2

            i -= 1

    @staticmethod
    def visualize_network(ffnn):
        G = nx.DiGraph()
        layer_labels = []
        positions = {}
        for i in range(len(ffnn.input[0][0])):
            G.add_node(f"Input {i+1}")
            positions[f"Input {i+1}"] = (0, -i)
            layer_labels.append(f"Input {i+1}")
        prev_layer_neurons = [f"Input {i+1}" for i in range(len(ffnn.input[0][0]))]
        x_pos = 1 
        for idx, layer in enumerate(ffnn.layers):
            current_layer_neurons = []
            bias_node = f"Bias {idx+1}"
            positions[bias_node] = (x_pos -0.5, layer.n_neurons / 3)  
            G.add_node(bias_node)
            layer_labels.append(bias_node)
            for n in range(layer.n_neurons):
                neuron_name = f"Layer {idx+1}\nNeuron {n+1}"
                G.add_node(neuron_name)
                current_layer_neurons.append(neuron_name)
                positions[neuron_name] = (x_pos, -n)
                layer_labels.append(neuron_name)
                weight_bias = layer.weight[0][n] 
                G.add_edge(bias_node, neuron_name, weight=f"{weight_bias:.2f}")
                for prev_neuron in prev_layer_neurons:
                    weight_idx = prev_layer_neurons.index(prev_neuron) + 1 
                    weight = layer.weight[weight_idx][n]
                    G.add_edge(prev_neuron, neuron_name, weight=f"{weight:.2f}")
            prev_layer_neurons = current_layer_neurons
            x_pos += 1  
        plt.figure(figsize=(12, 8))
        nx.draw(G, positions, with_labels=True, node_color='pink', node_size=1000, font_size=10, font_color='darkblue')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, font_color='red', label_pos=0.25)
        plt.title("Feed Forward Neural Network Visualization")
        plt.show()