from NN1 import NN1
from sklearn.model_selection import train_test_split
from nn_utils import run_GDNN_model, read_dataset

X, Y = read_dataset(180, 500)
dataset = train_test_split(X, Y, test_size=0.3, random_state=1)

epochs = 600
population_size = 10
generations = 10

run_GDNN_model(NN1, epochs, population_size, generations, dataset)

