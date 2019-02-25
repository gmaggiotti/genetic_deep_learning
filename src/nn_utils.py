import numpy as np
from sklearn.utils import shuffle
import os


def read_dataset(features, rows):
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../dataset/data-400.csv", delimiter=",", skiprows=1, usecols=range(1, features)) \
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


def run_GDNN_model(nn_class, epochs, population_size, generations, dataset):
    gen = {}
    best_n_children = 4
    train_x, train_y, test_x, test_y = dataset
    ## Generate a poblation of neural networks each trained from a random starting weigth
    ## ordered by the best performers (low error)
    init_pob = [nn_class(train_x, train_y, test_x, test_y, epochs) for i in range(population_size)]
    init_pob = sort_by_fittest([(nn.calc_accuracy(test_x, test_y), nn) for nn in init_pob], Type.accuracy)
    acc, acc_std = init_pob[0][1].calc_accuracy(test_x, test_y)
    print("600,{},{},{}".format(init_pob[0][1].get_error(), acc, acc_std))
    gen[0] = init_pob
    for x in range(1, generations):
        population = []
        for i in range(population_size):
            parent1 = gen[x - 1][np.random.randint(best_n_children)][1]
            parent2 = gen[x - 1][np.random.randint(best_n_children)][1]
            w_child = crossover(parent1, parent2)
            aux = nn_class(train_x, train_y, test_x, test_y, epochs, w_child)
            population += [tuple((aux.calc_accuracy(test_x, test_y), aux))]
        gen[x] = sort_by_fittest(population, Type.accuracy)
        net = gen[x][0][1]
        acc, acc_std = net.calc_accuracy(test_x, test_y)
        print("{},{},{},{}".format((x + 1) * epochs, net.get_error(), acc, acc_std))
        del population
