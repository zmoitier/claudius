from accoster import abs2
from numpy import pi
from scipy.integrate import quad
from scipy.special import hankel1, jv


def normL2_rad(k, m, func, ρ):
    n = len(sol.func)
    if n == 0:
        f = lambda r: abs2(func(m, r)) * r
        return quad(f, 1, ρ)[0]
    elif n == 1:
        return 0
    elif n == 2:
        δ = sol.radii[0]
        return 0
    else:
        exit("Unsopported number of function in sol.func")


def normL2_disk(k, sol):
    pass
