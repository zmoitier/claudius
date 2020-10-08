from scipy.special import hankel1, jv

from rot_inv_scattering import Solution


def solution(k, M):
    m = range(M + 1)
    return Solution(-jv(m, k) / hankel1(m, k), ())
