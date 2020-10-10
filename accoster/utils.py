from math import ceil

from numpy import amax, arange, e, exp, sin, sqrt, where
from scipy.special import jv


def εμ_to_n(ε, μ):
    return sqrt(ε * μ)


def εμ_to_η(ε, μ):
    return sqrt(-ε * μ)


def M_trunc(k, T):
    m = arange(ceil(16 + e * k * T / 2))
    JmkT = jv(m, k * T)
    I = where(JmkT >= 1e-8)[0]
    return I[-1]


def M_trunc_none(k, T, r, M):
    if (T is not None) and (M is not None):
        M = M_trunc(k, T)
        return max(M, T)
    elif T is None:
        return M
    elif M is None:
        return M_trunc(k, T)
    else:
        T = amax(r)
        return M_trunc(k, T)


def incident_field(k, c1, c2, coord):
    if coord == "xy":
        return exp(1j * k * c2)
    elif coord == "rθ":
        return exp(1j * k * c1 * sin(c2))
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
