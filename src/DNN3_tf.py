import numpy as np
from NN3_tf import NN3_tf
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset

X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=1)

epochs = 6000

dataset = train_x, train_y, test_x, test_y
nn3 = NN3_tf(dataset, epochs, print_step=600)
