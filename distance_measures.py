import numpy as np


def dist_eucledian(x1, x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))


def manhattan_distance(x1, x2):
    return np.sum(np.absolute(np.subtract(x1, x2)))