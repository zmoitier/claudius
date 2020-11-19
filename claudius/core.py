from dataclasses import astuple, dataclass
from sys import exit

from numpy import isscalar, ndarray, ndim, ones_like
from scipy.special import h1vp, hankel1, spherical_jn, spherical_yn


@dataclass(frozen=True)
class Problem:
    dim: int  # Dimension of the problem: 2 or 3
    pde: str  # Which PDE to use: Helmholtz or Maxwell
    inn_bdy: str  # Inner boundary: P, D, or N
    radii: tuple  # Radii of the layers
    eps_mu: tuple  # Functions giving ε and μ
    wavenum: float  # Wavenumber of the incident wave
    fun: tuple  # Functions that solve the ODE
    fun_der: tuple  # Derivative of fun

    def __iter__(self):
        return iter(astuple(self))


@dataclass(frozen=True)
class Solution(Problem):
    coeff: ndarray  # Coefficients of the series


def check_dim(dim):
    if dim not in (2, 3):
        exit(
            """Unsupported inn_bdy, the choices are:
    dim = 2 or dim = 3."""
        )


def check_pde(pde):
    if not (pde.startswith("H") or pde.startswith("M")):
        exit(
            """Unsupported pde, the choices are:
    pde = "Helmholtz" or pde = "Maxwell"."""
        )


def check_inn_bdy(inn_bdy):
    if not (
        inn_bdy.startswith("D") or inn_bdy.startswith("N") or inn_bdy.startswith("P")
    ):
        exit(
            """Unsupported inn_bdy, the choices are:
    inn_bdy = "D" for Dirichlet condition on the inner radii,
    inn_bdy = "N" for Neumann condition on the inner radii,
    inn_bdy = "P" for a penetrable obstacle."""
        )


def check_layer(nb_layer, nb_εμ, nb_func, nb_func_der):
    if nb_εμ is not nb_layer:
        exit(f"""len(eps_mu) = {nb_εμ} instead of {nb_layer}""")
    if nb_func is not nb_layer:
        exit(f"""len(fun) = {nb_func} instead of {nb_layer}""")
    if nb_func_der is not nb_layer:
        exit(f"""len(fun) = {nb_func_der} instead of {nb_layer}""")


def to_fct(num_or_fct, a, b):
    if isscalar(num_or_fct):
        return lambda r: num_or_fct * ones_like(r)

    value_left, value_right = num_or_fct(a), num_or_fct(b)
    if (ndim(value_left) == 0) and (ndim(value_right) == 0):
        return num_or_fct

    exit("eps_mu should be numbres of function that return numbres.")


def make_fct(eps_mu, radii):
    return tuple(
        (to_fct(ε, a, b), to_fct(μ, a, b))
        for (ε, μ), a, b in zip(eps_mu, radii[:-1], radii[1:])
    )


def add_hankel(dim, pde, k, func, func_der):
    if dim == 2:
        fun = (*func, lambda m, r: hankel1(m, k * r))
        fun_der = (*func_der, lambda m, r: k * h1vp(m, k * r))

    if dim == 3:
        if pde.startswith("H"):
            fun = (
                *func,
                lambda l, r: spherical_jn(l, k * r) + 1j * spherical_yn(l, k * r),
            )
            fun_der = (
                *func_der,
                lambda l, r: k
                * (
                    spherical_jn(l, k * r, derivative=True)
                    + 1j * spherical_yn(l, k * r, derivative=True)
                ),
            )

        if pde.startswith("M"):
            exit("Not done yet")

    return (fun, fun_der)


def create_probem(dim, pde, inn_bdy, radii, eps_mu, wavenum, func, func_der):
    check_dim(dim)
    check_pde(pde)
    check_inn_bdy(inn_bdy)

    if inn_bdy.startswith("P"):
        check_layer(len(radii), len(eps_mu), len(func), len(func_der))
        εμ = make_fct(eps_mu, (0, *radii))
    else:
        check_layer(len(radii) - 1, len(eps_mu), len(func), len(func_der))
        εμ = make_fct(eps_mu, radii)

    fun, fun_der = add_hankel(dim, pde, wavenum, func, func_der)

    return Problem(dim, pde, inn_bdy, radii, εμ, wavenum, fun, fun_der)
