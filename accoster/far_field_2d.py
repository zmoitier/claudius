from numpy import cos, exp, ones_like, pi, sqrt


def sc_far_field(k, sol, θ):
    if len(sol.coeff.shape) == 1:
        β = sol.coeff
    else:
        β = sol.coeff[:, -1]
    θπ2 = θ - pi / 2
    ff = β[0] * ones_like(θ, dtype=complex)
    for m, c in enumerate(β[1:], start=1):
        ff += c * 2 * cos(m * θπ2)
    return sqrt(2 / (pi * k)) * exp(-1j * pi / 4) * ff
