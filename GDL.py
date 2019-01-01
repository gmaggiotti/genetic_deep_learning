import numpy as np

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

def mate(wa, wb):
    rf = np.random.randint(2, size=(7, 1))
    rf_inv = abs(rf -1)
    wa = rf * wa
    wb = rf_inv * wb
    return wa + wb

#input data, each column represent a dif neuron
X = np.loadtxt("1NN/X.txt",delimiter=",")
max = np.matrix(X).max()
X = 2*X/float(max) - 1
#output, are the one-hot encoded labels
y = np.loadtxt("1NN/Y.txt",delimiter=",").reshape(X.__len__(),1)

epochs = 600

#synapses
w0a = 2*np.random.random((X.size/X.__len__(),1)) - 1
w0b = 2*np.random.random((X.size/X.__len__(),1)) - 1

# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.
for j in xrange(epochs):

    l1a = sigmoid(np.dot(X, w0a))
    l1b = sigmoid(np.dot(X, w0b))

    # Error back propagation of errors using the chain rule.
    l1a_error = y - l1a
    l1b_error = y - l1b
    if(j % 10) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error a: " + str(np.mean(np.abs(l1a_error))) + "Error b: " + str(np.mean(np.abs(l1b_error))))

    adjustmenta = l1a_error*sigmoid(l1a, deriv=True)
    adjustmentb = l1b_error*sigmoid(l1b, deriv=True)

    #update weights (no learning rate term)
    w0a += X.T.dot(adjustmenta)
    w0b += X.T.dot(adjustmentb)

print("------")

w0c = mate(w0a,w0b)

for j in xrange(epochs):

    l1c = sigmoid(np.dot(X, w0c))

    # Error back propagation of errors using the chain rule.
    l1c_error = y - l1c
    if(j % 10) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error a: " + str(np.mean(np.abs(l1c_error))))

    adjustmentc = l1c_error*sigmoid(l1c, deriv=True)

    #update weights (no learning rate term)
    w0c += X.T.dot(adjustmentc)




