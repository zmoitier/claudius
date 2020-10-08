from numpy import array
from numpy.linalg import solve
from scipy.special import h1vp, hankel1, jv, jvp


def coeff_scattering(εc, k, M, C, Cp):
    A = array(
        [
            [[C(m, k), -hankel1(m, k)], [Cp(m, k) / εc, -h1vp(m, k)]]
            for m in range(M + 1)
        ]
    )
    F = array([[jv(m, k), jvp(m, k)] for m in range(M + 1)])
    return solve(A, F)
