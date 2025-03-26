from ffnn import FFNN
from layers import Layers 
from activations import tanh, softmax, linear
import numpy as np

# Inisialisasi 
ffnn = FFNN(batch_size=1, learning_rate=0.01, epoch=1, verbose=1, loss_func='mse', weight_init='uniform')

hidden_layer1 = Layers(n_inputs=3, n_neurons=3, activ_func=linear)
hidden_layer2 = Layers(n_inputs=3, n_neurons=2, activ_func=linear)
output_layer = Layers(n_inputs=2, n_neurons=2, activ_func=linear)

ffnn.addHiddenLayer(hidden_layer1)
ffnn.addHiddenLayer(hidden_layer2)
ffnn.addHiddenLayer(output_layer)

# input data
X = [[0.1, 0.5, -0.3]] 
Y = [[1], [0]]  

ffnn.addInputOutput(X, Y)
ffnn.feedForward()

print("Feed forward end")

FFNN.visualize_network(ffnn)