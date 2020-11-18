from claudius import create_probem

from .base_fun import fun_cst, fun_cst_der


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
    return create_probem(3, "Helmholtz", inn_bdy, radii, εμc, k, fun, fun_der)
