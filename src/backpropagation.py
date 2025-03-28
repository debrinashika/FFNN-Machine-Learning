import numpy as np
import activations

# turunan fungsi aktivasi
def linearDerivative(net):
    return 1

def tanDerivative(net):
    return  1 - np.tanh(net) ** 2

def softmaxDerivative(net):
    net = np.array(net)

    softmax_output = np.exp(np.clip(net, -100, 100))   
    softmax_output /= np.sum(softmax_output, keepdims=True)

    softmax_jacobian = np.zeros((len(net), len(net)))
    
    for i in range(len(net)):
        for j in range(len(net)):
            if i == j:
                softmax_jacobian[i, j] = softmax_output[i] * (1 - softmax_output[j])
            else:
                softmax_jacobian[i, j] = -softmax_output[i] * softmax_output[j]
    return softmax_jacobian


def sigmoidDerivative(net):
    net = np.array(net)
    return net * (1 - net)

def reluDerivative(net):
    net = np.array(net)
    return np.where(net > 0, 1, 0)

def eluDerivative(net, a=1.0):
    net = np.array(net)
    return np.where(net > 0, 1, activations.elu(net, a) + a)

def swishDerivative(net):
    net = np.array(net)
    s = activations.swish(net)
    return s + (1 - s) * (1 / (1 + np.exp(-net)))

# turunan fungsi loss
def mseDerivative(o, target):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    target = np.array(target).T 

    return (o - target) * 2/batch_size

def binaryDerivative(o, target):
    o = np.array(o) 
    target = np.array(target).T 
    batch_size, num_neurons = o.shape
    return (o - target)/(o*(1-o)) * 1/batch_size

def categoricalDerivative(o, target):
    o = np.array(o) 
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
        net = np.array(net).T  
        delta = []
        for i in range(len(net)):
            jacobian = softmaxDerivative(net[i]) 
            delta.append(np.dot(loss[i], jacobian).tolist())

        return delta
    elif activFunc =="tanh":
        return loss * tanDerivative(net).T
    elif activFunc == "sigmoid":
        return loss * sigmoidDerivative(net).T
    elif activFunc == "relu":
        return loss * reluDerivative(net).T
    elif activFunc == "swish":
        return loss * swishDerivative(net).T
    elif activFunc == "elu":
        return loss * eluDerivative(net).T
    else:
        return loss * linearDerivative(net)


def hiddenLayer(w, net, delta, activFunc):
    if activFunc=="softmax":
        dot = np.dot(w, delta)
        delta = []
        for i in range(len(dot)):
            jacobian = softmaxDerivative(net[i]) 
            delta.append(np.dot(dot[i], jacobian).tolist())
        return delta

    elif activFunc =="tanh":
        return np.dot(w, delta) * tanDerivative(net)
    elif activFunc == "sigmoid":
        return np.dot(w, delta) * sigmoidDerivative(net)
    elif activFunc == "relu":
        return np.dot(w, delta) * reluDerivative(net)
    elif activFunc == "elu":
        return np.dot(w, delta) * eluDerivative(net)
    elif activFunc == "swish":
        return np.dot(w, delta) * swishDerivative(net)
    else:
        return np.dot(w, delta) * linearDerivative(net)
