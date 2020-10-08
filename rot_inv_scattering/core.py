from dataclasses import dataclass

import numba
from numpy import ndarray


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


@dataclass
class Solution:
    coeff: ndarray
    func: tuple
