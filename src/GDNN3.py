import numpy as np
from NN3 import NN3
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nn_utils import crossover, Type, sort_by_fittest
import os

size = 500


def read_dataset():
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../dataset/data-500.csv", delimiter=",", skiprows=1, usecols=range(1, 180))[0:size]
    neurons = dataset.shape[1] - 1
    X = dataset[:, 0:neurons]
    Y = dataset[:, neurons].reshape(X.__len__(), 1)
    Y[Y > 1] = 0
    max = 100  # np.matrix(X).max()
    ### Improving gradient descent through feature scaling
    X = 2 * X / float(max) - 1
    return shuffle(X, Y, random_state=1)


X, Y = read_dataset()
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=1)

epochs = 60
best_n_children = 4
population_size = 10
gen = {}
generations = 10

## Generate a poblation of neural networks each trained from a random starting weigth
## ordered by the best performers (low error)
init_pob = [NN3(train_x, train_y, test_x, test_y, epochs) for i in range(population_size)]
init_pob = sort_by_fittest([(nn.calc_accuracy(test_x,test_y), nn) for nn in init_pob], Type.error)
print("600,{}".format(init_pob[0][1].get_error()))
gen[0] = init_pob

for x in range(1, generations):
    population = []
    for i in range(population_size):
        parent1 = gen[x - 1][np.random.randint(best_n_children)][1]
        parent2 = gen[x - 1][np.random.randint(best_n_children)][1]
        w_child = crossover(parent1, parent2)
        aux = NN3(train_x, train_y, test_x, test_y, epochs, w_child)
        population += [tuple((aux.calc_accuracy(test_x, test_y), aux))]
    gen[x] = sort_by_fittest(population,Type.accuracy)
    net = gen[x][0][1]
    print("{},{},{}".format((x + 1) * epochs, net.get_error(), net.calc_accuracy(test_x, test_y)))
    del population