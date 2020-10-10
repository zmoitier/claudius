from scipy.special import hankel1, jv

from accoster import Solution


def solution(k, M):
    m = range(M + 1)
    return Solution(-jv(m, k) / hankel1(m, k), (), (1,))
