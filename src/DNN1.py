from NN1 import NN1
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset

X, Y = read_dataset(180, 500)
dataset = train_test_split(
    X, Y, test_size=0.3, random_state=1)

epochs = 6000
nn1 = NN1(dataset, epochs, print_step=600)
