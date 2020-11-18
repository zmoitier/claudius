from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp

from claudius import εμ_to_n, εμ_to_η


def fun_cst(eps, mu, wavenum):
    if (eps * mu).real > 0:
        nk = εμ_to_n(eps, mu) * wavenum

        def f(m, r):
            return jv(m, nk * r)

        def g(m, r):
            return yv(m, nk * r)

    else:
        ηk = εμ_to_η(eps, mu) * wavenum

        def f(m, r):
            return iv(m, ηk * r)

        def g(m, r):
            return kv(m, ηk * r)

    return (f, g)


def fun_cst_der(eps, mu, wavenum, p=1):
    if (eps * mu).real > 0:
        nk = εμ_to_n(eps, mu) * wavenum
        c = nk ** p

        def f(m, r):
            return c * jvp(m, nk * r, p)

        def g(m, r):
            return c * yvp(m, nk * r, p)

    else:
        ηk = εμ_to_η(eps, mu) * wavenum
        c = ηk ** p

        def f(m, r):
            return c * ivp(m, ηk * r, p)

        def g(m, r):
            return c * kvp(m, ηk * r, p)

    return (f, g)
