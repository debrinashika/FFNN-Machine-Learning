import numpy as np

# turunan fungsi aktivasi
def linearDerivative(net):
    return 1

def tanDerivative(net):
    return  1 - np.tanh(net) ** 2

def softmaxDerivative(net):
    net = np.array(net) 

    batch_size, num_neurons = net.shape
    softmax_jacobian = np.zeros((batch_size, num_neurons, num_neurons))
    
    softmax_output = np.exp(np.clip(net, -100, 100))   
    softmax_output /= np.sum(softmax_output, axis=1, keepdims=True)

    for b in range(batch_size):
        for i in range(num_neurons):
            for j in range(num_neurons):
                if i == j:
                    softmax_jacobian[b, i, j] = softmax_output[b, i] * (1 - softmax_output[b, j])
                else:
                    softmax_jacobian[b, i, j] = -softmax_output[b, i] * softmax_output[b, j]

    return softmax_jacobian

# turunan fungsi loss
def mseDerivative(o, target):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    target = np.array(target).T  # Pastikan target berbentuk array
    print(f"o.shape: {o.shape}, target.shape: {target.shape}")
    print(f"batch size: {batch_size}")
    return (o - target) * 2/batch_size

def binaryDerivative(o, target):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    return (o - target)/(o*(1-o)) * 1/batch_size

def categoricalDerivative(o, target):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    return -(target/o) * 1/batch_size
    
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
        diag_jacobian = np.diagonal(jacobian, axis1=1, axis2=2)  
        return loss * diag_jacobian.T 
    elif activFunc =="tanh":
        return loss * tanDerivative(net).T
    else:
        return loss * linearDerivative(net)


def hiddenLayer(w, net, delta, activFunc):
    if activFunc=="softmax":
        net = np.array(net) 
        sigma_delta_w = np.dot(w, delta)  
        return np.einsum("bij,jb->bi", softmaxDerivative(net), sigma_delta_w)
    elif activFunc =="tanh":
        return np.dot(w, delta) * tanDerivative(net)
    else:
        return np.dot(w, delta) * linearDerivative(net)
