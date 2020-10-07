from math import ceil

import numba
from numpy import arange, e, exp, sin, sqrt, where
from scipy.special import jv


@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)


def M_trunc(k, T):
    m = arange(ceil(16 + e * k * T / 2))
    JmkT = jv(m, k * T)
    I = where(JmkT >= 1e-8)[0]
    return I[-1]


def incident_field(k, x, y, coord):
    if coord == "xy":
        return exp(1j * k * y)
    elif coord == "rθ":
        return exp(1j * k * x * sin(y))
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
