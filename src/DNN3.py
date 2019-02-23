from NN3 import NN3
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset

X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=1)

epochs = 6000

nn3 = NN3(train_x, train_y, test_x, test_y, epochs, print_step=600)
