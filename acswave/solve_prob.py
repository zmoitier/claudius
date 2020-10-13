from sys import exit

from numpy import arange, array, zeros
from numpy.linalg import solve
from scipy.special import h1vp, hankel1, jv, jvp

from acswave import Solution


def _calc_F(n, vm, k, R):
    F = zeros((len(vm), n))
    F[:, -2] = jv(vm, k * R)
    F[:, -1] = k * jvp(vm, k * R)
    return F


def _calc_A_I(inn_bdy, n, vm, δ, εμ, k, fun, fun_der):
    A = zeros((len(vm), n, n), dtype=complex)

    if inn_bdy.startswith("N"):
        A[:, 0, 0] = fun_der[0][0](vm, δ[0])
        A[:, 0, 1] = fun_der[0][1](vm, δ[0])
    else:
        A[:, 0, 0] = fun[0][0](vm, δ[0])
        A[:, 0, 1] = fun[0][1](vm, δ[0])

    for i, r in enumerate(δ[1:-1]):
        j = 2 * i
        A[:, j + 1, j] = fun[i][0](vm, r)
        A[:, j + 1, j + 1] = fun[i][1](vm, r)
        A[:, j + 1, j + 2] = -fun[i + 1][0](vm, r)
        A[:, j + 1, j + 3] = -fun[i + 1][1](vm, r)

        A[:, j + 2, j] = fun_der[i][0](vm, r) / εμ[i][0](r)
        A[:, j + 2, j + 1] = fun_der[i][1](vm, r) / εμ[i][0](r)
        A[:, j + 2, j + 2] = -fun_der[i + 1][0](vm, r) / εμ[i + 1][0](r)
        A[:, j + 2, j + 3] = -fun_der[i + 1][1](vm, r) / εμ[i + 1][0](r)

    A[:, -2, -3] = fun[-1][0](vm, δ[-1])
    A[:, -2, -2] = fun[-1][1](vm, δ[-1])
    A[:, -2, -1] = -hankel1(vm, k * δ[-1])

    A[:, -1, -3] = fun_der[-2][0](vm, δ[-1]) / εμ[-2][0](δ[-1])
    A[:, -1, -2] = fun_der[-2][1](vm, δ[-1]) / εμ[-2][0](δ[-1])
    A[:, -1, -1] = -h1vp(vm, k * δ[-1])

    return A


def _calc_A_P(n, vm, δ, εμ, k, fun, fun_der):
    A = zeros((len(vm), n, n), dtype=complex)

    A[:, 0, 0] = fun[0](vm, δ[0])
    A[:, 0, 1] = -fun[1][0](vm, δ[0])
    A[:, 0, 2] = -fun[1][1](vm, δ[0])

    A[:, 1, 0] = fun_der[0](vm, δ[0]) / εμ[0][0](δ[0])
    A[:, 1, 1] = -fun_der[1][0](vm, δ[0]) / εμ[1][0](δ[0])
    A[:, 1, 2] = -fun_der[1][1](vm, δ[0]) / εμ[1][0](δ[0])

    for i, r in enumerate(δ[1:-1], start=1):
        j = 2 * i
        A[:, j, j - 1] = fun[i][0](vm, r)
        A[:, j, j] = fun[i][1](vm, r)
        A[:, j, j + 1] = -fun[i + 1][0](vm, r)
        A[:, j, j + 2] = -fun[i + 1][1](vm, r)

        A[:, j + 1, j - 1] = fun_der[i][0](vm, r) / εμ[i][0](r)
        A[:, j + 1, j] = fun_der[i][1](vm, r) / εμ[i][0](r)
        A[:, j + 1, j + 1] = -fun_der[i + 1][0](vm, r) / εμ[i + 1][0](r)
        A[:, j + 1, j + 2] = -fun_der[i + 1][1](vm, r) / εμ[i + 1][0](r)

    A[:, -2, -3] = fun[-1][0](vm, δ[-1])
    A[:, -2, -2] = fun[-1][1](vm, δ[-1])
    A[:, -2, -1] = -hankel1(vm, k * δ[-1])

    A[:, -1, -3] = fun_der[-1][0](vm, δ[-1]) / εμ[-1][0](δ[-1])
    A[:, -1, -2] = fun_der[-1][1](vm, δ[-1]) / εμ[-1][0](δ[-1])
    A[:, -1, -1] = -h1vp(vm, k * δ[-1])

    return A


def _solve_2HI(prob, vm):
    δ = array(prob.radii)

    if len(δ) == 1:
        if prob.inn_bdy.startswith("N"):
            return Solution(*prob, -jvp(vm, prob.k * δ[0]) / prob.fun_der[0](vm, δ[0]))
        else:
            return Solution(*prob, -jv(vm, prob.k * δ[0]) / prob.fun[0](vm, δ[0]))

    else:
        nbu = 2 * len(δ) - 1
        A = _calc_A_I(prob.inn_bdy, nbu, vm, δ, prob.εμ, prob.k, prob.fun, prob.fun_der)
        F = _calc_F(nbu, vm, prob.k, δ[-1])
        return Solution(*prob, solve(A, F))


def _solve_2HP(prob, vm):
    δ = array(prob.radii)
    nbu = 2 * len(δ)
    A = _calc_A_P(nbu, vm, δ, prob.εμ, prob.k, prob.fun, prob.fun_der)
    F = _calc_F(nbu, vm, prob.k, δ[-1])
    return Solution(*prob, solve(A, F))


def solve_prob(prob, M):
    m = arange(M + 1)
    if (
        (prob.dim == 2)
        and prob.pde.startswith("H")
        and (prob.inn_bdy.startswith("D") or prob.inn_bdy.startswith("N"))
    ):
        return _solve_2HI(prob, m)

    if (prob.dim == 2) and prob.pde.startswith("H") and prob.inn_bdy.startswith("P"):
        return _solve_2HP(prob, m)

    exit("Error in solve_prob")
