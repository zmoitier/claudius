from numpy import ones_like

from claudius import create_probem

from .base_fun import fun_cst, fun_cst_der


def create_problem_cst(inn_bdy, radii, eps_mu_cst, wavenum):
    k = wavenum
    εμc = tuple((ε, μ) for (ε, μ) in eps_mu_cst)

    if inn_bdy.startswith("P"):
        fun = (
            fun_cst(eps_mu_cst[0][0], eps_mu_cst[0][1], k)[0],
            *(fun_cst(ε, μ, k) for (ε, μ) in eps_mu_cst[1:]),
        )
        fun_der = (
            fun_cst_der(eps_mu_cst[0][0], eps_mu_cst[0][1], k)[0],
            *(fun_cst_der(ε, μ, k) for (ε, μ) in eps_mu_cst[1:]),
        )
    else:
        fun = tuple(fun_cst(ε, μ, k) for (ε, μ) in eps_mu_cst)
        fun_der = tuple(fun_cst_der(ε, μ, k) for (ε, μ) in eps_mu_cst)

    return create_probem(2, "Helmholtz", inn_bdy, radii, εμc, k, fun, fun_der)
