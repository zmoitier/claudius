from sys import exit

from numpy import arctan2, cos, hypot, sin, size, where, zeros_like
from scipy.special import hankel1, jv

from rot_inv_scattering import M_trunc, incident_field
from rot_inv_scattering.disk import (coeff_scattering, inner_field,
                                     inner_field_der)


def total_field_(M, C, αβ, kr, θ, I, E):
    ut = zeros_like(kr, dtype=complex)
    ut[I] = αβ[0, 0] * C(0, kr[I])
    ut[E] = αβ[0, 1] * hankel1(0, kr[E]) + jv(0, kr[E])
    tmp = zeros_like(kr, dtype=complex)
    for m in range(1, M + 1):
        tmp[I] = αβ[m, 0] * C(m, kr[I])
        tmp[E] = αβ[m, 1] * hankel1(m, kr[E]) + jv(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        ut += tmp
    return ut


def scattered_field_(M, C, αβ, kr, θ, I, E):
    us = zeros_like(kr, dtype=complex)
    us[I] = αβ[0, 0] * C(0, kr[I]) - jv(0, kr[I])
    us[E] = αβ[0, 1] * hankel1(0, kr[E])
    tmp = zeros_like(kr, dtype=complex)
    for m in range(1, M + 1):
        tmp[I] = αβ[m, 0] * C(m, kr[I]) - jv(m, kr[I])
        tmp[E] = αβ[m, 1] * hankel1(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        us += tmp
    return us


def total_field(εc, μc, k, T, x, y, coord):
    C, Cp = inner_field(εc, μc), inner_field_der(εc, μc)
    M = M_trunc(k, T)
    αβ = coeff_scattering(εc, k, M, C, Cp)

    if coord == "xy":
        r = hypot(x, y)
        θ = arctan2(y, x)
    elif coord == "rθ":
        r, θ = x, y
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
    kr = k * r
    I = where(r <= 1)
    E = where(r > 1)

    if size(I[0]) < size(E[0]):
        u_in = incident_field(k, x, y, coord)
        u_sc = scattered_field_(M, C, αβ, kr, θ, I, E)
        return u_in + u_sc
    else:
        return total_field_(M, C, αβ, kr, θ, I, E)


def scattered_field(εc, μc, k, T, x, y, coord):
    C, Cp = inner_field(εc, μc), inner_field_der(εc, μc)
    M = M_trunc(k, T)
    αβ = coeff_scattering(εc, k, M, C, Cp)

    if coord == "xy":
        r = hypot(x, y)
        θ = arctan2(y, x)
    elif coord == "rθ":
        r, θ = x, y
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
    kr = k * r
    I = where(r <= 1)
    E = where(r > 1)

    if size(I[0]) <= size(E[0]):
        return scattered_field_(M, C, αβ, kr, θ, I, E)
    else:
        u_in = incident_field(k, x, y, coord)
        u_to = total_field_(M, C, αβ, kr, θ, I, E)
        return u_to - u_in
