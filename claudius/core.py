"""Core class"""
from dataclasses import astuple, dataclass

from numpy import isscalar, ndarray, ndim, ones_like
from scipy.special import h1vp, hankel1, spherical_jn, spherical_yn


@dataclass(frozen=True)
class Problem:
    """
    Dataclass to describe the Problem

    Attributes
    ----------
    dim : integer
        Dimension of the problem: 2 or 3
    pde : string
        Which PDE to use: Helmholtz or Maxwell
    inn_bdy : string
        Which inner boundary: Penetrable, Dirichlet, or Neumann
    radii : tuple
        Radii of the layers
    eps_mu : tuple
        ε and μ of the layers
    wavenum : float
        Wavenumber of the incident wave
    fun : tuple
        Functions that solve the ODE
    fun_der : tuple
        Derivative of fun
    """

    dim: int
    pde: str
    inn_bdy: str
    radii: tuple
    eps_mu: tuple
    wavenum: float
    fun: tuple
    fun_der: tuple

    def __iter__(self):
        return iter(astuple(self))


@dataclass(frozen=True)
class Solution(Problem):
    """
    Dataclass to describe the solution that is an inheritance from Problem

    Attributes
    ----------
    coeff : ndarray
        Coefficients of the series solution
    """

    coeff: ndarray


def _check_dim(dim):
    if dim not in (2, 3):
        raise ValueError(f"Dimension {dim} unsupported, the choices are: 2 or 3.")


def _check_pde(pde):
    if not (pde.startswith("H") or pde.startswith("M")):
        raise ValueError(
            f"PDE {pde} unsupported, the choise are: 'Helmholtz' or 'Maxwell'."
        )


def _check_inn_bdy(inn_bdy):
    if not (
        inn_bdy.startswith("D") or inn_bdy.startswith("N") or inn_bdy.startswith("P")
    ):
        raise ValueError(
            f"Inner boundary {inn_bdy} unsupported, the choise are:\n"
            "    'Dirichlet' for Dirichlet condition on the inner radii,\n"
            "    'Neumann' for Neumann condition on the inner radii,\n"
            "    'Penetrable' for a penetrable obstacle."
        )


def _check_layer(nb_layer, nb_εμ, nb_func, nb_func_der):
    if nb_εμ is not nb_layer:
        raise ValueError(f"len(eps_mu) = {nb_εμ} instead of {nb_layer}")
    if nb_func is not nb_layer:
        raise ValueError(f"len(fun) = {nb_func} instead of {nb_layer}")
    if nb_func_der is not nb_layer:
        raise ValueError(f"len(fun) = {nb_func_der} instead of {nb_layer}")


def _to_fct(num_or_fct, a, b):
    if isscalar(num_or_fct):
        return lambda r: num_or_fct * ones_like(r)

    value_left, value_right = num_or_fct(a), num_or_fct(b)
    if (ndim(value_left) == 0) and (ndim(value_right) == 0):
        return num_or_fct

    raise ValueError("eps_mu should be numbers of function that return numbers.")


def _make_fct(eps_mu, radii):
    return tuple(
        (_to_fct(ε, a, b), _to_fct(μ, a, b))
        for (ε, μ), a, b in zip(eps_mu, radii[:-1], radii[1:])
    )


def _add_hankel(dim, pde, k, func, func_der):
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
            raise ValueError("Not implemented yet")

    return (fun, fun_der)


def create_probem(dim, pde, inn_bdy, radii, eps_mu, wavenum, func, func_der):
    """
    prob = create_probem(dim, pde, inn_bdy, radii, eps_mu, wavenum, func, func_der)

    Create the Problem dataclass from arguments.

    Parameters
    ----------
    dim : integer
        Dimension of the problem: 2 or 3
    pde : string
        Which PDE to use: Helmholtz or Maxwell
    inn_bdy : string
        Which inner boundary: Penetrable, Dirichlet, or Neumann
    radii : tuple
        Radii of the layers
    eps_mu : tuple
        ε and μ of the layers
    wavenum : float
        Wavenumber of the incident wave
    fun : tuple
        Functions that solve the ODE
    fun_der : tuple
        Derivative of fun

    Returns
    -------
    prob : Problem
        Problem dataclass created from arguments.
    """

    _check_dim(dim)
    _check_pde(pde)
    _check_inn_bdy(inn_bdy)

    if inn_bdy.startswith("P"):
        _check_layer(len(radii), len(eps_mu), len(func), len(func_der))
        εμ = _make_fct(eps_mu, (0, *radii))
    else:
        _check_layer(len(radii) - 1, len(eps_mu), len(func), len(func_der))
        εμ = _make_fct(eps_mu, radii)

    fun, fun_der = _add_hankel(dim, pde, wavenum, func, func_der)

    return Problem(dim, pde, inn_bdy, radii, εμ, wavenum, fun, fun_der)
