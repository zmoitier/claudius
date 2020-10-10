from numpy import amax

from accoster import M_trunc, sc_field, to_polar, tt_field
from accoster.ann_cts import solution


def scattered_field(δ, εc, μc, k, c1, c2, coord, T=None):
    r, θ = to_polar(c1, c2, coord)
    if T is None:
        T = amax(r)
    M = M_trunc(k, T)

    sol = solution(δ, εc, μc, k, M)
    return sc_field(k, sol, r, θ, "rθ")


def total_field(δ, εc, μc, k, c1, c2, coord, T=None):
    r, θ = to_polar(c1, c2, coord)
    if T is None:
        T = amax(r)
    M = M_trunc(k, T)

    sol = solution(δ, εc, μc, k, M)
    return tt_field(k, sol, r, θ, "rθ")
