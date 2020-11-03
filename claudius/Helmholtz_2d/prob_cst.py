from numpy import ones_like

from claudius import create_probem, solve_prob, to_polar, trunc_H2d, trunc_None

from .base_fun import fun_cst, fun_cst_der
from .calc_field import sc_field, tt_field


def create_problem_cst(inn_bdy, radii, εμ_cst, k):
    εμc = tuple(
        (lambda r: ε * ones_like(r), lambda r: μ * ones_like(r)) for (ε, μ) in εμ_cst
    )

    if inn_bdy.startswith("P"):
        fun = (
            fun_cst(εμ_cst[0][0], εμ_cst[0][1], k)[0],
            *(fun_cst(ε, μ, k) for (ε, μ) in εμ_cst[1:]),
        )
        fun_der = (
            fun_cst_der(εμ_cst[0][0], εμ_cst[0][1], k)[0],
            *(fun_cst_der(ε, μ, k) for (ε, μ) in εμ_cst[1:]),
        )
    else:
        fun = tuple(fun_cst(ε, μ, k) for (ε, μ) in εμ_cst)
        fun_der = tuple(fun_cst_der(ε, μ, k) for (ε, μ) in εμ_cst)

    return create_probem(2, "Helmholtz", inn_bdy, radii, εμc, k, fun, fun_der)


def scattered_field(prob, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)

    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return sc_field(sol, r, θ)


def total_field(prob, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)

    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return tt_field(sol, r, θ)
