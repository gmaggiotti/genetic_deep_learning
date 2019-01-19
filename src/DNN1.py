import numpy as np
from NN1 import NN1
from NN3 import NN3
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

size = 500


def read_dataset():
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../dataset/data-500.csv", delimiter=",", skiprows=1, usecols=range(1,180))[0:size]
    neurons = dataset.shape[1] - 1
    X = dataset[:, 0:neurons]
    Y = dataset[:, neurons].reshape(X.__len__(), 1)
    Y[Y > 1] = 0
    maxn = 100  # np.matrix(X).maxn()
    # Improving gradient descent through feature scaling
    X = 2 * X / float(maxn) - 1
    return shuffle(X, Y, random_state=1)


X, Y = read_dataset()
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.2, random_state=1)

epochs = 6000

nn1 = NN1(train_x, train_y, test_x, test_y, epochs, print_step=600)
