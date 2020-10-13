from sys import exit

from numpy import array, cos, isscalar, ones_like, sin, where, zeros_like
from scipy.special import hankel1

from acswave import incident_field, to_polar


def _partial_w_I(k, m, coeff, fun, I, R):
    u = zeros_like(R, dtype=complex)
    for j, J in enumerate(I[1:-1]):
        p = 2 * j
        q = p + 1
        u[J] = coeff[p] * fun[j][0](m, R[J]) + coeff[q] * fun[j][1](m, R[J])
    u[I[-1]] = coeff[-1] * hankel1(m, k * R[I[-1]])


def _partial_w_P(k, m, coeff, fun, I, R):
    u = zeros_like(R, dtype=complex)
    u[I[0]] = coeff[0] * fun[0](m, R[I[0]])
    for j, J in enumerate(I[1:-1], start=1):
        q = 2 * j
        p = q - 1
        u[J] = coeff[p] * fun[j][0](m, R[J]) + coeff[q] * fun[j][1](m, R[J])
    u[I[-1]] = coeff[-1] * hankel1(m, k * R[I[-1]])


def _partial_u_I(k, coeff, fun, I, R, Θ):
    u = _partial_w_I(k, 0, coeff[0, :], fun, I, R) * ones_like(Θ)
    for m in range(1, size(coeff, 0)):
        tmp = _partial_w_I(k, m, coeff[m, :], fun, I, R) * ones_like(Θ)
        if m % 2:
            tmp *= 2j * sin(m * Θ)
        else:
            tmp *= 2 * cos(m * Θ)
        u += tmp
    return u


def _partial_u_P(k, coeff, fun, I, R, Θ):
    u = _partial_w_P(k, 0, coeff[0, :], fun, I, R) * ones_like(Θ)
    for m in range(1, size(coeff, 0)):
        tmp = _partial_w_P(k, m, coeff[m, :], fun, I, R) * ones_like(Θ)
        if m % 2:
            tmp *= 2j * sin(m * Θ)
        else:
            tmp *= 2 * cos(m * Θ)
        u += tmp
    return u


def sc_field_2d(sol, r, θ):
    if sol.dim == 3:
        exit("Wrong dimension it should be 2 for this function.")

    R = r if not isscalar(r) else array([r])
    Θ = θ if not isscalar(θ) else array([θ])

    δ = sol.radii
    n = len(δ)
    I = (
        where(R < δ[0]),
        *(where(δ[i] <= R < δ[i + 1]) for i in range(n - 1)),
        where(δ[-1] <= R),
    )

    if sol.inn_bdy.startswith("P"):
        us = _partial_u_P(sol.k, sol.coeff, sol.fun, I, R, Θ)
    else:
        us = _partial_u_I(sol.k, sol.coeff, sol.fun, I, R, Θ)

    return us
    n = len(sol.func)
    if n == 0:
        E = where(r >= 1)
        us[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        return us
    elif n == 1:
        I = where(r < 1)
        E = where(r >= 1)
        us = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        us[I] -= incident_field(k, r[I], θ[I], "rθ")
        return us
    elif n == 2:
        δ = sol.radii[0]
        I = where(r <= δ)
        A = where((r > δ) & (r < 1))
        E = where(r >= 1)
        us = _partial_field_2(k, sol.coeff, sol.func[0], sol.func[1], r, θ, I, A, E)
        us[I] -= incident_field(k, r[I], θ[I], "rθ")
        us[A] -= incident_field(k, r[A], θ[A], "rθ")
        return us
    else:
        exit("Unsopported number of function in sol.func")


def tt_field_2d(k, sol, r, θ):
    ut = 1j * zeros_like(r) * zeros_like(θ)
    n = len(sol.func)
    if n == 0:
        E = where(r >= 1)
        ut[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        ut[E] += incident_field(k, r[E], θ[E], "rθ")
        return ut
    elif n == 1:
        I = where(r < 1)
        E = where(r >= 1)
        ut = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        ut[E] += incident_field(k, r[E], θ[E], "rθ")
        return ut
    elif n == 2:
        δ = sol.radii[0]
        I = where(r <= δ)
        A = where((r > δ) & (r < 1))
        E = where(r >= 1)
        ut = _partial_field_2(k, sol.coeff, sol.func[0], sol.func[1], r, θ, I, A, E)
        ut[E] += incident_field(k, r[E], θ[E], "rθ")
        return ut
    else:
        exit("Unsopported number of function in sol.func")
