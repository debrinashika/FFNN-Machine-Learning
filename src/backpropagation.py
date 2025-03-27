import numpy as np

# turunan fungsi aktivasi
def linearDerivative(net):
    return 1

def tanDerivative(net):
    return  1 - np.tanh(net) ** 2

import numpy as np

def softmaxDerivative(net, threshold=100):
    net = np.array(net)  
    batch_size, num_neurons = net.shape
  
    softmax_output = np.exp(np.clip(net, -100, 100))   
    softmax_output /= np.sum(softmax_output, axis=1, keepdims=True)

    if num_neurons > threshold:
        return softmax_output * (1 - softmax_output)

    softmax_jacobian = np.zeros((batch_size, num_neurons, num_neurons))
    
    for b in range(batch_size):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i == j:
                    softmax_jacobian[b, i, j] = softmax_output[b, i] * (1 - softmax_output[b, j])
                else:
                    softmax_jacobian[b, i, j] = -softmax_output[b, i] * softmax_output[b, j]

    return softmax_jacobian


def sigmoidDerivative(net):
    net = np.array(net)
    return net * (1 - net)

def reluDerivative(net):
    net = np.array(net)
    return np.where(net > 0, 1, 0)

# turunan fungsi loss
def mseDerivative(o, target):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    target = np.array(target).T 
    print(f"o.shape: {o.shape}, target.shape: {target.shape}")
    print(f"batch size: {batch_size}")
    return (o - target) * 2/batch_size

def binaryDerivative(o, target):
    o = np.array(o) 
    target = np.array(target).T 
    batch_size, num_neurons = o.shape
    return (o - target)/(o*(1-o)) * 1/batch_size

def categoricalDerivative(o, target):
    o = np.array(o) 
    target = np.array(target).T 
    target = np.array(target).T
    batch_size, num_neurons = o.shape
    return -(target / o) * (1 / batch_size)
    
# hitung gradien
def outputLayer(o, net, target, activFunc, LossFunc):
    if LossFunc == "binary":
        loss =  binaryDerivative(o, target) 
    elif LossFunc == "categorical":
        loss = categoricalDerivative(o, target) 
    else:
        loss =  mseDerivative(o, target) 
    
    if activFunc=="softmax":
        jacobian = softmaxDerivative(net) 
        if jacobian.ndim == 2:  
            diag_jacobian = jacobian  
        else:  
            diag_jacobian = np.diagonal(jacobian, axis1=1, axis2=2) 
        return loss * diag_jacobian.T 
    elif activFunc =="tanh":
        return loss * tanDerivative(net).T
    elif activFunc == "sigmoid":
        return loss * sigmoidDerivative(net).T
    elif activFunc == "relu":
        return loss * reluDerivative(net).T
    else:
        return loss * linearDerivative(net)


def hiddenLayer(w, net, delta, activFunc):
    if activFunc=="softmax":
        jacobian = softmaxDerivative(net) 
        if jacobian.ndim == 2:  
            diag_jacobian = jacobian  
        else:  
            diag_jacobian = np.diagonal(jacobian, axis1=1, axis2=2) 
        return np.dot(w, delta) * diag_jacobian
    elif activFunc =="tanh":
        return np.dot(w, delta) * tanDerivative(net)
    elif activFunc == "sigmoid":
        return np.dot(w, delta) * sigmoidDerivative(net)
    elif activFunc == "relu":
        return np.dot(w, delta) * reluDerivative(net)
    else:
        return np.dot(w, delta) * linearDerivative(net)
