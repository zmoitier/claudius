from numpy import amax

from rot_inv_scattering import M_trunc, sc_field, to_polar, tt_field
from rot_inv_scattering.disk_trans import solution


def scattered_field(εc, μc, k, c1, c2, coord, T=None):
    r, θ = to_polar(c1, c2, coord)
    if T is None:
        T = amax(r)
    M = M_trunc(k, T)

    sol = solution(εc, μc, k, M)
    return sc_field(k, sol, r, θ, "rθ")


def total_field(εc, μc, k, c1, c2, coord, T=None):
    r, θ = to_polar(c1, c2, coord)
    if T is None:
        T = amax(r)
    M = M_trunc(k, T)

    sol = solution(εc, μc, k, M)
    return tt_field(k, sol, r, θ, "rθ")
