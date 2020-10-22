from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp

from claudius import εμ_to_n, εμ_to_η


def CD_cst(ε, μ, k):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k
        return (lambda m, r: jv(m, nk * r), lambda m, r: yv(m, nk * r))
    else:
        ηk = εμ_to_η(ε, μ) * k
        return (lambda m, r: iv(m, ηk * r), lambda m, r: kv(m, ηk * r))


def CD_cst_der(ε, μ, k, p=1):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k
        c = nk ** p
        return (lambda m, r: c * jvp(m, nk * r, p), lambda m, r: c * yvp(m, nk * r, p))
    else:
        ηk = εμ_to_η(ε, μ) * k
        c = ηk ** p
        return (lambda m, r: c * ivp(m, ηk * r, p), lambda m, r: c * kvp(m, ηk * r, p))
