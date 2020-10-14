from sys import exit

from numpy import cos, ones_like, sin, where, zeros_like

from acswave import incident_field


def _partial_inner(α, C, R, Θ):
    u = α[0] * C(0, R) * ones_like(Θ)
    tmp = zeros_like(u)
    for m, a in enumerate(α[1:], start=1):
        tmp = a * C(m, R) * ones_like(Θ)
        if m % 2:
            tmp *= 2j * sin(m * Θ)
        else:
            tmp *= 2 * cos(m * Θ)
        u += tmp
    return u


def _partial_trans(α, C, β, D, R, Θ):
    u = (α[0] * C(0, R) + β[0] * D(0, R)) * ones_like(Θ)
    tmp = zeros_like(u)
    for m, (a, b) in enumerate(zip(α[1:], β[1:]), start=1):
        tmp = (a * C(0, R) + b * D(0, R)) * ones_like(Θ)
        if m % 2:
            tmp *= 2j * sin(m * Θ)
        else:
            tmp *= 2 * cos(m * Θ)
        u += tmp
    return u


def _partial_outer(β, H, R, Θ):
    u = β[0] * H(0, R) * ones_like(Θ)
    tmp = zeros_like(u)
    for m, b in enumerate(β[1:], start=1):
        tmp = b * H(m, R) * ones_like(Θ)
        if m % 2:
            tmp *= 2j * sin(m * Θ)
        else:
            tmp *= 2 * cos(m * Θ)
        u += tmp
    return u


def sc_field(sol, R, Θ):
    if sol.dim == 3:
        exit("Wrong dimension it should be 2 for this function.")

    δ = sol.radii
    n = len(δ)
    Inn = where(R < δ[0])
    Lay = (where((δ[i] <= R) & (R < δ[i + 1])) for i in range(n - 1))
    Out = where(δ[-1] <= R)

    us = 1j * zeros_like(R) * zeros_like(Θ)

    if sol.inn_bdy.startswith("P") and (len(Inn[0]) != 0):
        us[Inn] = _partial_inner(
            sol.coeff[:, 0], sol.fun[0], R[Inn], Θ[Inn]
        ) - incident_field(sol.k, R[Inn], Θ[Inn], "rθ")

    shift = 1 if sol.inn_bdy.startswith("P") else 0
    for i, J in enumerate(Lay):
        if len(J[0]) != 0:
            us[J] = (
                _partial_trans(
                    sol.coeff[:, 2 * i + shift],
                    sol.fun[i + shift][0],
                    sol.coeff[:, 2 * i + shift + 1],
                    sol.fun[i + shift][1],
                    R[J],
                    Θ[J],
                )
                - incident_field(sol.k, R[J], Θ[J], "rθ")
            )

    if len(Out[0]) != 0:
        us[Out] = _partial_outer(sol.coeff[:, -1], sol.fun[-1], R[Out], Θ[Out])

    return us


def tt_field(sol, R, Θ):
    if sol.dim == 3:
        exit("Wrong dimension it should be 2 for this function.")

    δ = sol.radii
    n = len(δ)
    Inn = where(R < δ[0])
    Lay = (where((δ[i] <= R) & (R < δ[i + 1])) for i in range(n - 1))
    Out = where(δ[-1] <= R)

    ut = 1j * zeros_like(R) * zeros_like(Θ)

    if sol.inn_bdy.startswith("P") and (len(Inn[0]) != 0):
        ut[Inn] = _partial_inner(sol.coeff[:, 0], sol.fun[0], R[Inn], Θ[Inn])

    shift = 1 if sol.inn_bdy.startswith("P") else 0
    for i, J in enumerate(Lay):
        if len(J[0]) != 0:
            ut[J] = _partial_trans(
                sol.coeff[:, 2 * i + shift],
                sol.fun[i + shift][0],
                sol.coeff[:, 2 * i + shift + 1],
                sol.fun[i + shift][1],
                R[J],
                Θ[J],
            )

    if len(Out[0]) != 0:
        ut[Out] = _partial_outer(
            sol.coeff[:, -1], sol.fun[-1], R[Out], Θ[Out]
        ) + incident_field(sol.k, R[Out], Θ[Out], "rθ")

    return ut
