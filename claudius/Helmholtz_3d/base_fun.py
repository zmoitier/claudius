from scipy.special import (spherical_in, spherical_jn, spherical_kn,
                           spherical_yn)

from claudius import εμ_to_n, εμ_to_η


def fun_cst(eps, mu, wavenum):
    if (eps * mu).real > 0:
        nk = εμ_to_n(eps, mu) * wavenum

        def f(l, r):
            return spherical_jn(l, nk * r)

        def g(l, r):
            return spherical_yn(l, nk * r)

    else:
        ηk = εμ_to_η(eps, mu) * wavenum

        def f(l, r):
            return spherical_in(l, ηk * r)

        def g(l, r):
            return spherical_kn(l, ηk * r)

    return (f, g)


def fun_cst_der(eps, mu, wavenum, p=1):
    if (eps * mu).real > 0:
        nk = εμ_to_n(eps, mu) * wavenum
        c = nk ** p

        def f(l, r):
            return c * spherical_jn(l, nk * r, derivative=True)

        def g(l, r):
            return c * spherical_yn(l, nk * r, derivative=True)

    else:
        ηk = εμ_to_η(eps, mu) * wavenum
        c = ηk ** p

        def f(l, r):
            return c * spherical_in(l, ηk * r, derivative=True)

        def g(l, r):
            return c * spherical_kn(l, ηk * r, derivative=True)

    return (f, g)
