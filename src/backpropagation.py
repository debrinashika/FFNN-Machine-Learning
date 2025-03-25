import numpy as np

# masih rusak
def softmaxOutput(o, target):
    o = np.array(o) 
    target = np.array(target)
    batch_size, num_neurons = o.shape
    J = np.zeros((num_neurons, num_neurons))
    
    for i in range(num_neurons):
        for j in range(num_neurons):
            for b in range(batch_size):
                J[i, j] = o[b, i] * (1 - o[b, j]) 

    return J @ (-target / o)  

def tanOutput(o, net, target):
    target = np.array(target)  
    turunan = 1 - np.tanh(net) ** 2
    return (o - target) * turunan

# masih rusak
def softmaxHidden(w, o, delta):
    o = np.array(o) 
    batch_size, num_neurons = o.shape
    J = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            for b in range(batch_size):
                J[i, j] = o[b, i] * (1 - o[b, j]) 

    print("ini weight", w)
    print("ini delta",delta)
    print("ini j",J)

    return np.dot(w,delta.T) 

def tanHidden(w, o, delta):
    # print("JAJAJ")
    # print(delta)
    # print(w)
    return np.dot(w, delta) * o  


def linearOutput(o, target, batch_size):
    o = np.array(o) 
    target = np.array(target)
    print(f"Shape: {len(o)}")
    if batch_size >= 2:
        target = target.flatten()
    
    #turunannya selalu 1
    return (o - target) * 1

def linearHidden(w, delta):
    return np.dot(w, delta)
