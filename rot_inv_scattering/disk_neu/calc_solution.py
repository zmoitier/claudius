from scipy.special import h1vp, jvp

from rot_inv_scattering import M_trunc, Solution


def solution(k, T):
    m = range(M_trunc(k, T) + 1)
    return Solution(-jvp(m, k) / h1vp(m, k), ())
