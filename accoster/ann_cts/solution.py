from numpy import array
from numpy.linalg import solve
from scipy.special import h1vp, hankel1, jv, jvp

from accoster import Solution
from accoster.ann_cts import inner_field, inner_field_der


def solution(δ, εc, μc, k, M):
    C, D = inner_field(εc, μc)
    Cp, Dp = inner_field_der(εc, μc)

    m = range(M + 1)
    kδ = k * δ

    Jδ0, Jδ1 = -jv(m, kδ), -jvp(m, kδ)
    Cδ0, Cδ1 = C(m, kδ), Cp(m, kδ) / εc
    Dδ0, Dδ1 = D(m, kδ), Dp(m, kδ) / εc

    C0, C1 = C(m, k), Cp(m, k) / εc
    D0, D1 = D(m, k), Dp(m, k) / εc
    H0, H1 = -hankel1(m, k), -h1vp(m, k)
    J0, J1 = jv(m, k), jvp(m, k)

    A = array(
        [
            [
                [Jδ0[m], Cδ0[m], Dδ0[m], 0],
                [Jδ1[m], Cδ1[m], Dδ1[m], 0],
                [0, C0[m], D0[m], H0[m]],
                [0, C1[m], D1[m], H1[m]],
            ]
            for m in range(M + 1)
        ]
    )
    F = array([[0, 0, J0[m], J1[m]] for m in range(M + 1)])

    return Solution(solve(A, F), (C, D), (δ, 1))
