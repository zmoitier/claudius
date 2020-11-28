from math import ceil, sqrt

import numpy as np
from scipy.special import sph_harm, spherical_jn


def trunc_H3d(k, T):
    l = np.arange(ceil(16 + k * T))
    I = np.where(
        np.abs(np.sqrt((2 * l + 1) / (4 * np.pi)) * spherical_jn(l, k * T)) > 1e-6
    )
    return I[0][-1]


def incident_field(k, z):
    return np.exp(1j * k * z)


def calc_inc_field(k, r, θ, φ, L):
    u = 1j * np.zeros_like(r) * np.zeros_like(θ) * np.zeros_like(φ)
    for l in range(L):
        u += (
            1j ** l
            * sqrt(4 * np.pi * (2 * l + 1))
            * spherical_jn(l, k * r)
            * sph_harm(0, l, θ, φ)
        )
    return u


k = 10
L = trunc_H3d(k, 3)
print(f"L = {L}")

r = np.linspace(0.5, 3, num=16)
θ = np.linspace(0, 2 * np.pi, num=16, endpoint=False)
φ = np.linspace(0, np.pi, num=18)[1:-1]
R, Θ, Φ = np.meshgrid(r, θ, φ)
Z = R * np.cos(Φ)

print(np.amax(np.abs(incident_field(k, Z) - calc_inc_field(k, R, Θ, Φ, L))))
