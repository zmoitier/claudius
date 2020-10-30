from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp

from claudius import εμ_to_n, εμ_to_η


def fun_cst(ε, μ, k):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k

        def f(m, r):
            return jv(m, nk * r)

        def g(m, r):
            return yv(m, nk * r)

    else:
        ηk = εμ_to_η(ε, μ) * k

        def f(m, r):
            return iv(m, ηk * r)

        def g(m, r):
            return kv(m, ηk * r)

    return (f, g)


def fun_cst_der(ε, μ, k, p=1):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k
        c = nk ** p

        def f(m, r):
            return c * jvp(m, nk * r, p)

        def g(m, r):
            return c * yvp(m, nk * r, p)

    else:
        ηk = εμ_to_η(ε, μ) * k
        c = ηk ** p

        def f(m, r):
            return c * ivp(m, ηk * r, p)

        def g(m, r):
            return c * kvp(m, ηk * r, p)

    return (f, g)
