from numpy import cos, exp, ones_like, pi, sin, sqrt, where, zeros_like
from scipy.special import sph_harm

from .plane_wave import incident_field


def _partial_inner(α, C, R, Theta, Phi):
    u = α[0] * C(0, R) * sph_harm(0, 0, Theta, Phi)
    for l, a in enumerate(α[1:], start=1):
        u += a * C(l, R) * sph_harm(0, l, Theta, Phi)
    return u


def _partial_trans(α, C, β, D, R, Theta, Phi):
    u = (α[0] * C(0, R) + β[0] * D(0, R)) * sph_harm(0, 0, Theta, Phi)
    for l, (a, b) in enumerate(zip(α[1:], β[1:]), start=1):
        u += (a * C(l, R) + b * D(l, R)) * sph_harm(0, l, Theta, Phi)
    return u


def _partial_outer(β, H, R, Theta, Phi):
    u = β[0] * H(0, R) * sph_harm(0, 0, Theta, Phi)
    for l, b in enumerate(β[1:], start=1):
        u += b * H(l, R) * sph_harm(0, l, Theta, Phi)
    return u


def sc_field(sol, R, Theta, Phi):
    ρ = sol.radii
    N = len(ρ) - 1
    Inn = where(R < ρ[0])
    Lay = (where((ρ[i] <= R) & (R < ρ[i + 1])) for i in range(N))
    Out = where(ρ[-1] <= R)

    us = 1j * zeros_like(R) * zeros_like(Theta) * zeros_like(Phi)

    if sol.inn_bdy.startswith("P") and (len(Inn[0]) != 0):
        us[Inn] = _partial_inner(
            sol.coeff[:, 0], sol.fun[0], R[Inn], Theta[Inn], Phi[Inn]
        ) - incident_field(sol.wavenum, R[Inn], Theta[Inn], Phi[Inn], "spherical")

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
                    Theta[J],
                    Phi[J],
                )
                - incident_field(sol.wavenum, R[J], Theta[J], Phi[J], "spherical")
            )

    if len(Out[0]) != 0:
        us[Out] = _partial_outer(
            sol.coeff[:, -1], sol.fun[-1], R[Out], Theta[Out], Phi[Out]
        )

    return us


def tt_field(sol, R, Theta, Phi):
    ρ = sol.radii
    N = len(ρ) - 1
    Inn = where(R < ρ[0])
    Lay = (where((ρ[i] <= R) & (R < ρ[i + 1])) for i in range(N))
    Out = where(ρ[-1] <= R)

    ut = 1j * zeros_like(R) * zeros_like(Theta) * zeros_like(Phi)

    if sol.inn_bdy.startswith("P") and (len(Inn[0]) != 0):
        ut[Inn] = _partial_inner(
            sol.coeff[:, 0], sol.fun[0], R[Inn], Theta[Inn], Phi[Inn]
        )

    shift = 1 if sol.inn_bdy.startswith("P") else 0
    for i, J in enumerate(Lay):
        if len(J[0]) != 0:
            ut[J] = _partial_trans(
                sol.coeff[:, 2 * i + shift],
                sol.fun[i + shift][0],
                sol.coeff[:, 2 * i + shift + 1],
                sol.fun[i + shift][1],
                R[J],
                Theta[J],
                Phi[J],
            )

    if len(Out[0]) != 0:
        ut[Out] = _partial_outer(
            sol.coeff[:, -1], sol.fun[-1], R[Out], Theta[Out], Phi[Out]
        ) + incident_field(sol.wavenum, R[Out], Theta[Out], Phi[Out], "spherical")

    return ut


def f_field(sol, Theta, Phi):
    k = sol.wavenum
    β = sol.coeff[:, -1]

    ff = β[0] * sph_harm(0, 0, Theta, Phi)
    for l, c in enumerate(β[1:], start=1):
        ff += c * (-1j) ** l * sph_harm(0, l, Theta, Phi)
    return -1j * ff / k
