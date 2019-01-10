import numpy as np
import os

learning_rate = 0.001


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class NN1:
    def __init__(self, train_x, train_y, test_x, test_y, epochs, w=None, print_step=None):
        self.l1_error = 0
        if w is None:
            self.w0 = 2 * np.random.random((train_x.size / train_x.__len__(), 1)) - 1
        else:
            self.w0 = w
        for j in xrange(1, epochs + 1):
            l1 = sigmoid(np.dot(train_x, self.w0))
            self.l1_error = train_y - l1

            if (print_step is not None) and ((j % print_step == 0) or j == epochs):
                accuracy = self.calc_accuracy(test_x, test_y)
                print("{},{},{}".format(j, np.mean(np.abs(self.l1_error)), accuracy))

            adjustment = self.l1_error * sigmoid(l1, deriv=True)
            self.w0 += train_x.T.dot(adjustment) * learning_rate

    def get_weight(self):
        return self.w0

    def get_error(self):
        return np.mean(np.abs(self.l1_error))

    def calc_accuracy(self, test_x, test_y):
        prime_y = sigmoid(np.dot(test_x, self.w0))
        y_error = test_y - prime_y
        return 1 - np.mean(np.abs(y_error))
