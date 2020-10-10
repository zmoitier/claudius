from sys import exit

from numpy import cos, ones_like, sin, where, zeros_like
from scipy.special import hankel1, jv

from accoster import incident_field, to_polar


def _partial_field_0(k, β, r, θ):
    kr = k * r

    us = β[0] * hankel1(0, kr) * ones_like(θ)
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
    w0 = zeros_like(kr, dtype=complex)
    w0[I] = αβ[0, 0] * C(0, kr[I])
    w0[E] = αβ[0, 1] * hankel1(0, kr[E])

    u = w0 * ones_like(θ)
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


def _partial_field_2(k, abcd, C, D, r, θ, I, A, E):
    kr = k * r
    w0 = zeros_like(kr, dtype=complex)
    w0[I] = abcd[0, 0] * jv(0, kr[I])
    w0[A] = abcd[0, 1] * C(0, kr[A]) + abcd[0, 2] * D(0, kr[A])
    w0[E] = abcd[0, 3] * hankel1(0, kr[E])

    u = w0 * ones_like(θ)
    tmp = zeros_like(u)
    for m, c in enumerate(abcd[1:], start=1):
        tmp[I] = c[0] * jv(m, kr[I])
        tmp[A] = c[1] * C(m, kr[A]) + c[2] * D(m, kr[A])
        tmp[E] = c[3] * hankel1(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        u += tmp
    return u


def sc_field(k, sol, c1, c2, coord):
    r, θ = to_polar(c1, c2, coord)
    us = zeros_like(c1, dtype=complex) * zeros_like(c2, dtype=complex)
    n = len(sol.func)
    if n == 0:
        E = where(r >= 1)
        us[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        return us
    elif n == 1:
        I = where(r < 1)
        E = where(r >= 1)
        us = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        us[I] -= incident_field(k, c1[I], c2[I], coord)
        return us
    elif n == 2:
        δ = sol.radii[0]
        I = where(r <= δ)
        A = where((r > δ) & (r < 1))
        E = where(r >= 1)
        us = _partial_field_2(k, sol.coeff, sol.func[0], sol.func[1], r, θ, I, A, E)
        us[I] -= incident_field(k, c1[I], c2[I], coord)
        us[A] -= incident_field(k, c1[A], c2[A], coord)
        return us
    else:
        exit("Unsopported number of function in sol.func")


def tt_field(k, sol, c1, c2, coord):
    r, θ = to_polar(c1, c2, coord)
    ut = zeros_like(c1, dtype=complex) * zeros_like(c2, dtype=complex)
    n = len(sol.func)
    if n == 0:
        E = where(r >= 1)
        ut[E] = _partial_field_0(k, sol.coeff, r[E], θ[E])
        ut[E] += incident_field(k, c1[E], c2[E], coord)
        return ut
    elif n == 1:
        I = where(r < 1)
        E = where(r >= 1)
        ut = _partial_field_1(k, sol.coeff, sol.func[0], r, θ, I, E)
        ut[E] += incident_field(k, c1[E], c2[E], coord)
        return ut
    elif n == 2:
        δ = sol.radii[0]
        I = where(r <= δ)
        A = where((r > δ) & (r < 1))
        E = where(r >= 1)
        ut = _partial_field_2(k, sol.coeff, sol.func[0], sol.func[1], r, θ, I, A, E)
        ut[E] += incident_field(k, c1[E], c2[E], coord)
        return ut
    else:
        exit("Unsopported number of function in sol.func")
