import numpy as np
def enum(**enums):
    return type('Enum', (), enums)


Type = enum(error=1, accuracy=2)


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def mate(wa, wb):
    rf = np.random.randint(2, size=(wa.size, 1))
    rf_inv = abs(rf - 1)
    wa = rf * wa
    wb = rf_inv * wb
    return wa + wb

def sort_by_fittest(population, type):
    if type == Type.error :
        return sorted(population)
    else:
        return sorted(population, reverse=True)
