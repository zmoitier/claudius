from accoster import Solution
from scipy.special import hankel1, jv


def solution(k, M):
    m = range(M + 1)
    return Solution(-jv(m, k) / hankel1(m, k), (), (1,))
