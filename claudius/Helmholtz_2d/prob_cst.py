from numpy import ones_like

from claudius import create_probem

from .base_fun import fun_cst, fun_cst_der


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
