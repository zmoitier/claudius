from math import ceil
from sys import exit

from numba import complex64, complex128, float32, float64, vectorize
from numpy import absolute, amax, arange, pi, sqrt, where
from scipy.special import jv, spherical_jn


@vectorize([float64(complex128), float32(complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)


def trunc_H2d(k, T):
    m = arange(ceil(16 + k * T))
    I = where(absolute(jv(m, k * T)) > 1e-6)
    return I[0][-1]


def trunc_H3d(k, T):
    l = arange(ceil(16 + k * T))
    I = where(absolute(sqrt(4 * pi * (2 * l + 1)) * spherical_jn(l, k * T)) > 1e-6)
    return I[0][-1]


def trunc_None(trunc, k, T, r, M):
    if (T is not None) and (M is not None):
        exit("Specify either T or M but not both")

    if T is not None:
        return trunc(k, T)

    if M is not None:
        return M

    return trunc(k, amax(r))
