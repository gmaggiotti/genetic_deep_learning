import numpy as np
from sklearn.utils import shuffle
import os


def read_dataset(features, rows):
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../dataset/data-500.csv", delimiter=",", skiprows=1, usecols=range(1, features))\
        [0:rows]
    neurons = dataset.shape[1] - 1
    X = dataset[:, 0:neurons]
    Y = dataset[:, neurons].reshape(X.__len__(), 1)
    Y[Y > 1] = 0
    # Improving gradient descent through feature scaling
    # X = 2 * X / np.amax(X,0) - 1
    X = 2 * X / float(100) - 1
    return shuffle(X, Y, random_state=1)


def enum(**enums):
    return type('Enum', (), enums)


Type = enum(error=1, accuracy=2)


def sigmoid(x, deriv=False):
    if (deriv):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def crossover(wa, wb):
    result = []
    wa_weights = wa.get_weights()
    wb_weights = wb.get_weights()

    for i in range(0, len(wa_weights)):
        rf = np.random.randint(2, size=(wa_weights[i].shape[0], wa_weights[i].shape[1]))
        rf_inv = abs(rf - 1)
        wa_weights[i] = rf * wa_weights[i]
        wb_weights[i] = rf_inv * wb_weights[i]
        result.append(wa_weights[i] + wb_weights[i])
    return result


def sort_by_fittest(population, type):
    if type == Type.error:
        return sorted(population)
    else:
        return sorted(population, reverse=True)
