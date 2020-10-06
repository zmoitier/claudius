from sys import exit

from numpy import arctan2, array, cos, hypot, sin, where, zeros, zeros_like
from numpy.linalg import solve
from scipy.special import h1vp, hankel1, jv, jvp

from rot_inv_scattering import M_trunc
from rot_inv_scattering.disk import inner_field, inner_field_der


def coeff_scattering(εc, k, M, C, Cp):
    A = array(
        [
            [[C(m, k), -hankel1(m, k)], [Cp(m, k) / εc, -h1vp(m, k)]]
            for m in range(M + 1)
        ]
    )
    F = array([[jv(m, k), jvp(m, k)] for m in range(M + 1)])
    return solve(A, F)


def total_field(εc, μc, k, T, x, y, coord="xy"):
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

    ut = zeros_like(x, dtype=complex)
    ut[I] = αβ[0, 0] * C(0, kr[I])
    ut[E] = αβ[0, 1] * hankel1(0, kr[E]) + jv(0, kr[E])
    tmp = zeros_like(x, dtype=complex)
    for m in range(1, M + 1):
        tmp[I] = αβ[m, 0] * C(m, kr[I])
        tmp[E] = αβ[m, 1] * hankel1(m, kr[E]) + jv(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        ut += tmp
    return ut


def scattered_field(εc, μc, k, T, x, y, coord="xy"):
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

    ut = zeros_like(x, dtype=complex)
    ut[I] = αβ[0, 0] * C(0, kr[I]) - jv(0, kr[I])
    ut[E] = αβ[0, 1] * hankel1(0, kr[E])
    tmp = zeros_like(x, dtype=complex)
    for m in range(1, M + 1):
        tmp[I] = αβ[m, 0] * C(m, kr[I]) - jv(m, kr[I])
        tmp[E] = αβ[m, 1] * hankel1(m, kr[E])
        if m % 2:
            tmp *= 2j * sin(m * θ)
        else:
            tmp *= 2 * cos(m * θ)
        ut += tmp
    return ut
