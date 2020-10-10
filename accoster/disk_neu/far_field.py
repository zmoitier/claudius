from numpy import amax

from accoster import M_trunc, sc_far_field
from accoster.disk_neu import solution


def far_field(k, θ, M=None):
    if M is None:
        M = M_trunc(k, 2)

    sol = solution(k, M)
    return sc_far_field(k, sol, θ)
