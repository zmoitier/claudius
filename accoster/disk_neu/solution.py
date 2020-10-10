from scipy.special import h1vp, jvp

from accoster import Solution


def solution(k, M):
    m = range(M + 1)
    return Solution(-jvp(m, k) / h1vp(m, k), (), (1,))
