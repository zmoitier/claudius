from accoster import M_trunc, sc_far_field
from accoster.ann_cts import solution
from numpy import amax, pi


def far_field(δ, εc, μc, k, θ, M=None):
    if M is None:
        M = M_trunc(k, 4 * pi / k)

    sol = solution(δ, εc, μc, k, M)
    return sc_far_field(k, sol, θ)
