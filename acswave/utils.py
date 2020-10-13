from math import ceil
from sys import exit

from numba import complex64, complex128, float32, float64, vectorize
from numpy import abs, amax, arange, e, exp, sin, sqrt, where
from scipy.special import jv


@vectorize([float64(complex128), float32(complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)


def M_trunc_2d(k, T):
    m = arange(ceil(16 + e * k * T / 2))
    I = where(abs(jv(m, k * T)) >= 1e-8)[0]
    return I[-1]


def M_none_2d(k, T, r, M):
    if (T is not None) and (M is not None):
        exit("Specify either T or M but not both")
    elif T is not None:
        return M_trunc_2d(k, T)
    elif M is not None:
        return M
    else:
        return M_trunc_2d(k, amax(r))


def incident_field(k, c1, c2, coord):
    if coord == "xy":
        return exp(1j * k * c2)
    elif coord == "rθ":
        return exp(1j * k * c1 * sin(c2))
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
