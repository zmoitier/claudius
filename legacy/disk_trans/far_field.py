from accoster import M_trunc, sc_far_field
from accoster.disk_trans import solution
from numpy import amax, pi


def far_field(εc, μc, k, θ, M=None):
    if M is None:
        M = M_trunc(k, 4 * pi / k)

    sol = solution(εc, μc, k, M)
    return sc_far_field(k, sol, θ)
