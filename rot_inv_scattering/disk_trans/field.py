from scipy.special import iv, ivp, jv, jvp

from rot_inv_scattering import εμ_to_n, εμ_to_η


def inner_field(ε, μ):
    if (ε * μ).real > 0:
        n = εμ_to_n(ε, μ)
        return lambda m, r: jv(m, n * r)
    else:
        η = εμ_to_η(ε, μ)
        return lambda m, r: iv(m, η * r)


def inner_field_der(ε, μ, p=1):
    if (ε * μ).real > 0:
        n = εμ_to_n(ε, μ)
        c = n ** p
        return lambda m, z: c * jvp(m, n * z, p)
    else:
        η = εμ_to_η(ε, μ)
        c = η ** p
        return lambda m, z: c * ivp(m, η * z, p)
