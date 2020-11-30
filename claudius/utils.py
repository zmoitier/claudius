from numba import complex64, complex128, float32, float64, vectorize
from numpy import sqrt


@vectorize([float64(complex128), float32(complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)
