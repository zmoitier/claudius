from sys import exit

from numpy import cos, nan, sin, where, zeros_like
from scipy.special import hankel1

from rot_inv_scattering import incident_field, to_polar


def coeff_field_0(k, β, c1, c2, coord):
    r, θ = to_polar(c1, c2, coord)
    I = where(r < 1)
    E = where(r >= 1)

    kr = k * r
    us = zeros_like(r, dtype=complex)
    us[I] = 0
    us[E] = β[0] * hankel1(0, kr[E])
    tmp = zeros_like(us)
    for m, c in enumerate(β[1:], start=1):
        tmp[E] = c * hankel1(m, kr[E])
        if m % 2:
            tmp[E] *= 2j * sin(m * θ[E])
        else:
            tmp[E] *= 2 * cos(m * θ[E])
        us[E] += tmp[E]
    return us


def scattered_field(k, sol, c1, c2, coord):
    n = len(sol.func)
    if n == 0:
        return coeff_field_0(k, sol.coeff, c1, c2, coord)
    elif n == 1:
        pass
    elif n == 2:
        pass
    else:
        exit("Unsopported number of function in sol.func")


def total_field(k, sol, c1, c2, coord):
    n = len(sol.func)
    if n == 0:
        r, θ = to_polar(c1, c2, coord)
        I = where(r < 1)
        E = where(r >= 1)

        us = scattered_field(k, sol, r[E], θ[E], "rθ")
        ui = incident_field(k, c1[E], c2[E], coord)

        ut = zeros_like(r, dtype=complex)
        ut[I] = 0
        ut[E] = ui + us
        return ut
    elif n == 1:
        pass
    elif n == 2:
        pass
    else:
        exit("Unsopported number of function in sol.func")
