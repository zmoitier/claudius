from numpy import amax

from accoster import M_trunc_none, sc_field, to_polar, tt_field
from accoster.ann_cts import solution


def scattered_field(δ, εc, μc, k, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)
    M = M_trunc_none(k, T, r, M)

    sol = solution(δ, εc, μc, k, M)
    return sc_field(k, sol, r, θ)


def total_field(δ, εc, μc, k, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)
    M = M_trunc_none(k, T, r, M)

    sol = solution(δ, εc, μc, k, M)
    return tt_field(k, sol, r, θ)
