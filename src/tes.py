from ffnn import FFNN
from layers import Layers 
from activations import tanh, softmax, linear
import numpy as np

# Inisialisasi 
ffnn = FFNN(batch_size=1, learning_rate=0.5, epoch=10, verbose=1, loss_func='categorical', weight_init='custom')

hidden_layer1 = Layers(n_inputs=2, n_neurons=2, activ_func=tanh)
output_layer = Layers(n_inputs=2, n_neurons=2, activ_func=tanh)

# hidden_layer1 = Layers(n_inputs=2, n_neurons=2, activ_func=linear)
# hidden_layer2 = Layers(n_inputs=3, n_neurons=2, activ_func=linear)
# output_layer = Layers(n_inputs=2, n_neurons=2, activ_func=linear)

ffnn.addHiddenLayer(hidden_layer1)
ffnn.addHiddenLayer(output_layer)

# input data
X = [[0.05, 0.1]] 
Y = [[0.01], [0.99]]  

ffnn.addInputOutput(X, Y)
ffnn.feedForward()

print("Feed forward end")

# ffnn.plot_weight_distribution()
# ffnn.plot_gradient_distribution()

# FFNN.visualize_network(ffnn)
# ilustrasi graph dalam kasus ini biar ga bingung
# X = [[0.1, 0.5, -0.3], [0.7, -0.2, 0.4]] 
# -> idx menggambarkan input untuk batch misal yg [0.1, 0.5, -0.3] utk batch 1

# Layers[0] -> simpan info bobot buat di layer 1
# [[ 0.41200571 -0.34086336  0.23482862]
#  [-0.15228948  0.3455872   0.1827585 ]
# [-0.30291392 -0.28685229 -0.39861129]
# [-0.28425849 -0.11571413  0.45606305]]
# karena inputnya 3 + 1 bias maka ada 4 x 3 bobot, hidden layer 1 nerima 4 input jadi punya 4 bobot
# ini urutannya bobot bias, i1, i2, i3  terus kan bias = [ 0.41200571 -0.34086336  0.23482862] itu isinya weight buat ke neuron 1 2 dan 3

# ini hasil input tranpose prinsipnya sama kek yg lain
# [[[1.0, 1.0], [0.1, 0.7], [0.5, -0.2], [-0.3, 0.4]], [[1.0, 1.0], [0.3190574210281599, 0.24706301339055028], [-0.3927238515428944, -0.08764206918530601], [-0.08282988249334541, 0.5545353542326257]], [[1.0, 1.0], [0.45469907811963467, 0.5453009218803654], [0.533085554321989, 0.46691444567801105]]]
# idx 0 buat input ke layer 1 (input ke semua neuron sama yang beda itu bobotnya)
# urutannya buat udx 1 itu bias, i1, i2, i3

# ini buat net
# [[[0.3305973508123411, 0.2522824676079339], [-0.4150165433129566, -0.08786750584211604], [-0.08302009223035695, 0.6249070435895331]], [[-0.011056174051705503, 0.17064578509044787], [-0.1302348572171493, -0.262770742124488]], [[-0.6816679521507398, -0.691627110687776]]]
# ini hasil input setelah di aktivasi karena dia neuronnya 3 berarti kan outputnya 3 juga jadi di idx 0 net ada 3 list 
# yang isinya output di layer 1 (output neuron 1, neuron 2, neuron 3)
