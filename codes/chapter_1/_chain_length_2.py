from typing import List

Array_function = callable[[ndarray], ndarray]

Chain = List[Array_function]

def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    """
    Evaluates two functions in a row, in a "Chain".
    """

    assert len(chain) == 2, \
    "Length of input 'chain' should be 2"

    f1 = chain[0]
    f2 = chain[1]

    return f2(f1(x))

