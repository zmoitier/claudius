from scipy.special import (spherical_in, spherical_jn, spherical_kn,
                           spherical_yn)

from claudius import εμ_to_n, εμ_to_η


def fun_cst(ε, μ, k):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k

        def f(l, r):
            return spherical_jn(l, nk * r)

        def g(l, r):
            return spherical_yn(l, nk * r)

    else:
        ηk = εμ_to_η(ε, μ) * k

        def f(l, r):
            return spherical_in(l, ηk * r)

        def g(l, r):
            return spherical_kn(l, ηk * r)

    return (f, g)


def fun_cst_der(ε, μ, k, p=1):
    if (ε * μ).real > 0:
        nk = εμ_to_n(ε, μ) * k
        c = nk ** p

        def f(l, r):
            return c * spherical_jn(l, nk * r, derivative=True)

        def g(l, r):
            return c * spherical_yn(l, nk * r, derivative=True)

    else:
        ηk = εμ_to_η(ε, μ) * k
        c = ηk ** p

        def f(l, r):
            return c * spherical_in(l, ηk * r, derivative=True)

        def g(l, r):
            return c * spherical_kn(l, ηk * r, derivative=True)

    return (f, g)
