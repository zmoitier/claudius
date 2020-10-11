from numpy import array
from numpy.linalg import solve
from scipy.special import h1vp, hankel1, jv, jvp

from accoster import Solution
from accoster.disk_trans import inner_field, inner_field_der


def solution(εc, μc, k, M):
    C, Cp = inner_field(εc, μc), inner_field_der(εc, μc)

    m = range(M + 1)
    C0, C1 = C(m, k), Cp(m, k) / εc
    H0, H1 = -hankel1(m, k), -h1vp(m, k)
    J0, J1 = jv(m, k), jvp(m, k)

    A = array([[[C0[m], H0[m]], [C1[m], H1[m]]] for m in range(M + 1)])
    F = array([[J0[m], J1[m]] for m in range(M + 1)])

    return Solution(solve(A, F), (C,), (1,))
