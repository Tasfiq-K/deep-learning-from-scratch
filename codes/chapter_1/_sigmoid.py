import numpy as np
from numpy import ndarray


def sigmoid(x: ndarray) -> ndarray:
    """ 
    Apply the sigmoid function to each element in the input ndarray
    """

    return 1 / (1 + np.exp(-x))



arr = np.array([2.5, 6, 3.5])

print(sigmoid(arr))