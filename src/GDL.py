import numpy as np
from NN import NN


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def mate(wa, wb):
    rf = np.random.randint(2, size=(7, 1))
    rf_inv = abs(rf - 1)
    wa = rf * wa
    wb = rf_inv * wb
    return wa + wb


# input data, each column represent a dif neuron
X = np.loadtxt("../dataset/X.txt", delimiter=",")
max = np.matrix(X).max()
X = 2 * X / float(max) - 1
# output, are the one-hot encoded labels
y = np.loadtxt("../dataset/Y.txt", delimiter=",").reshape(X.__len__(), 1)

epochs=6000
best_n_children = 4
population_size = 10
gen ={}
generations = 3


## Generate a poblation of neural networks each trained from a random starting weigth
## ordered by the best performers (low error)
init_pob = [NN(X, y,epochs) for i in range(10)]
init_pob = sorted([(nn.get_error(), nn) for nn in init_pob])

gen[0]=init_pob
print "---"

for x in range(1,generations):
    population = []
    print("--- generation {} ---".format(x))
    for i in range(population_size):
        candidate1 = init_pob[np.random.randint(best_n_children)][1].get_weight()
        candidate2 = init_pob[np.random.randint(best_n_children)][1].get_weight()
        w_child = mate(candidate1, candidate2)
        aux = NN(X, y, epochs, w_child)
        population += [tuple((aux.get_error(), aux))]
    gen[x]=sorted(population)
    del population

print 'EoF'
NN(X,y,6000)
