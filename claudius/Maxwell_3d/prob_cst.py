from sys import exit

from claudius import (Problem, Solution, create_probem, solve_prob, trunc_H2d,
                      trunc_None)

from .base_fun import fun_cst, fun_cst_der
from .calc_field import sc_field, tt_field


def create_problem_cst(inn_bdy, radii, εμc, k):
    N = len(radii) - 1
    if inn_bdy.startswith("P"):
        fun = (fun_cst(*εμc[0], k)[0], *(fun_cst(*εμc[n], k) for n in range(1, N)))
        fun_der = (
            fun_cst_der(*εμc[0], k)[0],
            *(fun_cst_der(*εμc[n], k) for n in range(1, N)),
        )
    else:
        fun = tuple(fun_cst(εμc[n][0], εμc[n][1], k) for n in range(N))
        fun_der = tuple(fun_cst_der(εμc[n][0], εμc[n][1], k) for n in range(N))
    return create_probem(2, "Helmholtz", inn_bdy, radii, εμc, k, fun, fun_der)


def scattered_field(prob, r, θ, T=None, M=None):
    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return sc_field(sol, r, θ)


def total_field(prob, r, θ, T=None, M=None):
    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return tt_field(sol, r, θ)
