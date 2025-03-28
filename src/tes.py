from ffnn import FFNN
from layers import Layers 
from activations import tanh, softmax, linear, sigmoid, relu, elu, swish
import numpy as np

ffnn = FFNN(batch_size=4, learning_rate=0.5, epoch=5, verbose=1, loss_func='mse', weight_init='he', seed=42)

hidden_layer1 = Layers(n_inputs=2, n_neurons=3, activ_func=swish)
output_layer = Layers(n_inputs=3, n_neurons=2, activ_func=swish)

ffnn.addHiddenLayer(hidden_layer1)
ffnn.addHiddenLayer(output_layer)

X_train = [
    [0.05, 0.1], 
    [0.1, 0.2], 
    [0.2, 0.3], 
    [0.3, 0.4]
]

X_val = [
    [0.06, 0.12], 
    [0.15, 0.25], 
    [0.22, 0.33], 
    [0.35, 0.45]
]

Y_train = [
    [0.01, 0.99], 
    [0.05, 0.95], 
    [0.10, 0.90], 
    [0.15, 0.85]
]

Y_val = [
    [0.02, 0.98], 
    [0.06, 0.94], 
    [0.11, 0.89], 
    [0.16, 0.84]
]

ffnn.addInputTarget(X_train, X_val, Y_train, Y_val)

ffnn.feedForward()
# FFNN.visualize_network(ffnn)
print("Feed forward pertama selesai")

ffnn.feedForward()

# ffnn.plot_weight_distribution()
# ffnn.plot_gradient_distribution()
