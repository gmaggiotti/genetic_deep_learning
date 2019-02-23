import numpy as np
from NN3_tf import NN3_tf
from sklearn.model_selection import train_test_split
from nn_utils import crossover, Type, sort_by_fittest, read_dataset

X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.3, random_state=1)



X, Y = read_dataset(180, 500)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=1)

epochs = 600
best_n_children = 4
population_size = 20
gen = {}
generations = 10

## Generate a poblation of neural networks each trained from a random starting weigth
## ordered by the best performers (low error)
init_pob = [NN3_tf(train_x, train_y, test_x, test_y, epochs) for i in range(population_size)]
init_pob = sort_by_fittest([(nn.get_acc(), nn) for nn in init_pob], Type.accuracy)
print("600,{},{}".format(init_pob[0][1].get_error(),init_pob[0][1].get_acc()))

gen[0] = init_pob

for x in range(1, generations):
    population = []
    for i in range(population_size):
        parent1 = gen[x - 1][np.random.randint(best_n_children)][1]
        parent2 = gen[x - 1][np.random.randint(best_n_children)][1]
        w_child = crossover(parent1, parent2)
        aux = NN3_tf(train_x, train_y, test_x, test_y, epochs, w_child)
        population += [tuple((aux.get_acc(), aux))]
    gen[x] = sort_by_fittest(population, Type.accuracy)
    net = gen[x][0][1]
    print("{},{},{}".format((x + 1) * epochs, net.get_error(), net.get_acc()))
    del population
