from sys import exit

from numpy import cos, sin, where, zeros_like
from scipy.special import hankel1

from rot_inv_scattering import incident_field, to_polar


def _partial_field_0(k, β, r, θ):
    kr = k * r
    us = zeros_like(r, dtype=complex)
    us = β[0] * hankel1(0, kr)
    tmp = zeros_like(us)
    for m, c in enumerate(β[1:], start=1):
        tmp = c * hankel1(m, kr)
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        us += tmp
    return us


def _partial_field_1(k, αβ, C, r, θ, I, E):
    kr = k * r
    u = zeros_like(kr, dtype=complex)
    u[I] = αβ[0, 0] * C(0, kr[I])
    u[E] = αβ[0, 1] * hankel1(0, kr[E])
    tmp = zeros_like(u)
    for m, c in enumerate(αβ[1:], start=1):
        tmp[I] = c[0] * C(m, kr[I])
        tmp[E] = c[1] * hankel1(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        u += tmp
    return u


def sc_field(k, sol, c1, c2, coord):
    r, θ = to_polar(c1, c2, coord)
    I = where(r < 1)
    E = where(r >= 1)

    us = zeros_like(c1, dtype=complex)

    n = len(sol.func)
    if n == 0:
        us[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        return us
    elif n == 1:
        us = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        us[I] -= incident_field(k, c1[I], c2[I], coord)
        return us
    elif n == 2:
        pass
    else:
        exit("Unsopported number of function in sol.func")


def tt_field(k, sol, c1, c2, coord):
    r, θ = to_polar(c1, c2, coord)
    I = where(r < 1)
    E = where(r >= 1)

    ut = zeros_like(c1, dtype=complex)

    n = len(sol.func)
    if n == 0:
        ut[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        ut[E] += incident_field(k, c1[E], c2[E], coord)
        return ut
    elif n == 1:
        ut = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        ut[E] += incident_field(k, c1[E], c2[E], coord)
        return ut
    elif n == 2:
        pass
    else:
        exit("Unsopported number of function in sol.func")
