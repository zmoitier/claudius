from dataclasses import astuple, dataclass
from sys import exit

from numpy import ndarray
from scipy.special import h1vp, hankel1


@dataclass(frozen=True)
class Problem:
    dim: int
    pde: str
    inn_bdy: str
    radii: tuple
    εμ: tuple
    k: float
    fun: tuple
    fun_der: tuple

    def __iter__(self):
        return iter(astuple(self))


def create_probem(dim, pde, inn_bdy, radii, εμ, k, func, func_der):
    if not (
        ((2 <= dim <= 3) and pde.startswith("H"))
        or ((dim == 3) and pde.startswith("M"))
    ):
        exit(
            """Unsupported combinaison, the choices are:
    dim = 2 and pde = "Helmholtz",
    dim = 3 and pde = "Helmholtz",
    dim = 3 and pde = "Maxwell"."""
        )

    if not (
        inn_bdy.startswith("D") or inn_bdy.startswith("N") or inn_bdy.startswith("P")
    ):
        exit(
            """Unsupported inn_bdy, the choices are:
    inn_bdy = "D" for Dirichlet condition on the inner radii,
    inn_bdy = "N" for Neumann condition on the inner radii,
    inn_bdy = "P" for a penetrable obstacle."""
        )

    if inn_bdy.startswith("D") or inn_bdy.startswith("N"):
        if not ((len(radii) - 1) == len(εμ) == len(func) == len(func_der)):
            exit(
                """The lenght of radii -1 should be equal to the lenght of  εμ, func, and func_der."""
            )

    if inn_bdy.startswith("P"):
        if not (len(radii) == len(εμ) == len(func) == len(func_der)):
            exit("""The lenght of radii, εμ, func, and func_der should be equal.""")

    if dim == 2:
        fun = (*func, lambda m, r: hankel1(m, k * r))
        fun_der = (*func_der, lambda m, r: k * h1vp(m, k * r))

    if dim == 3:
        if pde.startswith("H"):
            pass

        if pde.startswith("M"):
            pass

    return Problem(dim, pde, inn_bdy, radii, εμ, k, fun, fun_der)


@dataclass(frozen=True)
class Solution(Problem):
    coeff: ndarray
