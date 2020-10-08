from scipy.special import hankel1, jv

from rot_inv_scattering import M_trunc, Solution


def solution(k, T):
    m = range(M_trunc(k, T) + 1)
    return Solution(-jv(m, k) / hankel1(m, k), ())
