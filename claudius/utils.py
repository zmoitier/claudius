from math import ceil
from sys import exit

from numba import complex64, complex128, float32, float64, vectorize
from numpy import absolute, amax, arange, e, exp, sin, sqrt, where
from scipy.special import jv, spherical_jn


@vectorize([float64(complex128), float32(complex64)])
def abs2(x):
    return x.real ** 2 + x.imag ** 2


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)


def trunc_H2d(k, T):
    m = arange(ceil(16 + e * k * T / 2))
    I = where(absolute(jv(m, k * T)) >= 1e-6)[0]
    return I[-1]


def trunc_H3d(k, T):
    m = arange(ceil(16.5 + e * k * T / 2))
    I = where(absolute(spherical_jn(m, k * T)) >= 1e-6)[0]
    return I[-1]


def trunc_None(trunc, k, T, r, M):
    if (T is not None) and (M is not None):
        exit("Specify either T or M but not both")
    elif T is not None:
        return trunc(k, T)
    elif M is not None:
        return M
    else:
        return trunc(k, amax(r))


def incident_field(k, c1, c2, coord):
    """Compute the plane wave incident field"""

    if coord == "xy":
        return exp(1j * k * c2)

    if coord == "rθ":
        return exp(1j * k * c1 * sin(c2))

    exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
