from dataclasses import astuple, dataclass
from sys import exit

from numpy import ndarray


@dataclass
class Problem:
    dim: int
    pde: str
    type: str
    radii: tuple
    εμ: tuple
    k: float
    fun: tuple
    fun_der: tuple

    def __iter__(self):
        return iter(astuple(self))


def create_probem(dim, pde, type, radii, εμ, k, fun, fun_der):
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

    if not (type.startswith("D") or type.startswith("N") or type.startswith("P")):
        exit(
            """Unsupported type, the choices are:
    type = "D" for Dirichlet condition on the inner radii,
    type = "N" for Neumann condition on the inner radii,
    type = "P" for a penetrable obstacle."""
        )

    if type.startswith("D") or type.startswith("N"):
        if not ((len(radii) - 1) == len(εμ) == len(fun) == len(fun_der)):
            exit(
                """The lenght of radii -1 should be equal to the lenght of  εμ, func, and func_der."""
            )

    if type.startswith("P"):
        if not (len(radii) == len(εμ) == len(fun) == len(fun_der)):
            exit("""The lenght of radii, εμ, func, and func_der should be equal.""")

    return Problem(dim, pde, type, radii, εμ, k, fun, fun_der)


@dataclass
class Solution(Problem):
    coeff: ndarray
