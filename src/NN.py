import numpy as np
import os

epochs = 60

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

class NN:

    def __init__(self, X, y, w=None):
        self.l1_error = 0
        if(w == None ):
            self.w0 = 2*np.random.random((X.size/X.__len__(),1)) - 1
        else:
            self.w0 = w
        for j in xrange(epochs):
            l1 = sigmoid(np.dot(X, self.w0))
            self.l1_error = y - l1

            if(j % 100) == 0:   # Only print the error every 10000 steps.
                print("Error: " + str(np.mean(np.abs(self.l1_error))))

            adjustment = self.l1_error*sigmoid(l1, deriv=True)
            self.w0 += X.T.dot(adjustment)

    def get_weight(self):
        return self.w0

    def get_error(self):
        return np.mean(np.abs(self.l1_error))