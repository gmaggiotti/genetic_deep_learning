from NN3 import NN3
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset

X, Y = read_dataset(180, 500)
dataset = train_test_split(
    X, Y, test_size=0.3, random_state=1)

epochs = 6000
nn3 = NN3(dataset, epochs, print_step=600)
