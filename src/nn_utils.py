import numpy as np


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
