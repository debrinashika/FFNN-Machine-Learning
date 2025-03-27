import numpy as np
from layers import Layers 
import activations 
from backpropagation import outputLayer,hiddenLayer
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import sys

class FFNN:
    def __init__(self, batch_size: int, learning_rate: float, epoch: int, verbose: int, loss_func, weight_init, seed=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.verbose = verbose
        self.loss_func = loss_func
        self.weight_init = weight_init  
        self.seed = seed  
        self.loss_train_history = []
        self.loss_val_history = []

        self.input_train: list[list[float]] = []
        self.input_val: list[list[float]] = []
        self.target_train: list[list[float]] = []
        self.target_val: list[list[float]] = []
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


    def addInputTarget(self, input_train: list[float], input_val: list[float], target_train: list[float], target_val: list[float]):
        self.input_train.append(input_train)
        self.input_val.append(input_val)
        self.target_train.append(target_train)
        self.target_val.append(target_val)

    def addHiddenLayer(self, layer: Layers):
        self.layers.append(layer)

    def plot_weight_distribution(self, layers_to_plot: list[int] = None):
        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.layers))) 

        for i in layers_to_plot:
            if i < 0 or i >= len(self.layers):
                print(f"Layer {i+1} tidak valid.")
                continue 
            
            weights = self.layers[i].weight.flatten() 
            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=30, alpha=0.7, color='b', edgecolor='black')
            plt.title(f'Distribusi Bobot - Layer {i+1}')
            plt.xlabel('Nilai Bobot')
            plt.ylabel('Frekuensi')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

    def plot_gradient_distribution(self, layers_to_plot: list[int] = None):
        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.delta_gradien)))  
        
        for i in layers_to_plot:
            if i < 0 or i >= len(self.delta_gradien):
                print(f"Layer {i+1} tidak valid.")
                continue  
            
            grad = self.delta_gradien[i].flatten()  
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
            train_error = 0
            
            for i, batch in enumerate(self.input_train):
                inputs: list[list[float]] = []
                nets: list[list[float]] = []
                self.initDeltaGradien()

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

                train_error += self.calcLoss(current, self.target_train[i]) 
       
                self.backPropagation(inputs, nets, self.target_train[i])
                self.updateWeight()
                print("\n")

            self.loss_train_history.append(train_error)

            val_error = 0
            val_pred = self.predict(self.input_val)
            val_error = self.calcLoss(val_pred, self.target_val) 
            self.loss_val_history.append(val_error)

            # **Progress Bar**
            if self.verbose == 1:
                sys.stdout.write(f"\rEpoch {j+1}/{self.epoch} - Training Loss: {train_error:.4f}  - Validation Loss: {val_error:.4f}")
                sys.stdout.flush()


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


    def save_model(self, filename):
        model_data = {
            "layers": [layer.weight for layer in self.layers], 
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epoch": self.epoch,
            "loss_func": self.loss_func
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model berhasil disimpan ke {filename}")

    def load_model(self, filename):
        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        if len(model_data["layers"]) != len(self.layers):
            raise ValueError("Jumlah layer pada model yang dimuat tidak sesuai.")

        for layer, weight in zip(self.layers, model_data["layers"]):
            layer.weight = weight

        self.batch_size = model_data["batch_size"]
        self.learning_rate = model_data["learning_rate"]
        self.epoch = model_data["epoch"]
        self.loss_func = model_data["loss_func"]

        print(f"Model berhasil dimuat dari {filename}")

    def plot_loss(self):
        print("Training Loss History:", self.loss_train_history)
        print("Validation Loss History:", self.loss_val_history)
        plt.plot(self.loss_train_history, label='Training Loss', color='blue')
        plt.plot(self.loss_val_history, label='Validation Loss', color='red', linestyle='dashed')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


    def predict(self, X):
        batch = np.array(X)  
        if batch.ndim == 3:  
            batch = batch.reshape(batch.shape[1], batch.shape[2])

        batch_size = batch.shape[0]
        bias = np.ones((batch_size, 1))
        print("Bias shape:", bias.shape)
        print("Batch shape before hstack:", batch.shape)
        current = np.hstack((bias, batch))  
        
        for layer in self.layers:
            net = np.dot(current, layer.weight) 
            current = layer.activ_func(net)
            if layer != self.layers[-1]:
                bias = np.ones((batch_size, 1))
                current = np.hstack((bias, current))  
        return current