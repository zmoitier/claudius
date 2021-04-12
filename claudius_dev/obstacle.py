""" Obstacle class """
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .constant import Constant


@dataclass(frozen=True)
class Obstacle:
    """
    Dataclass to describe the obstacle

    Attributes
    ----------
    shape : str
        Shape the obstacle: Disk or Sphere
    core : str
        Which Penetrable, Dirichlet, or Neumann
    radii : Tuple[float]
        Tuple of radii
    sig_rho : Tuple[
            Tuple[Callable[[np.array], np.array], ...],
            Tuple[Callable[[np.array], np.array], ...],
        ]
        σ and ρ of the layers
    """

    shape: str
    core: str
    radii: Tuple[float]
    sig_rho: Tuple[
        Tuple[Callable[[np.array], np.array], ...],
        Tuple[Callable[[np.array], np.array], ...],
    ]


def _check_shape(shape):
    shape = shape.capitalize()
    if shape not in ["Disk", "Sphere"]:
        raise ValueError(
            f"Shape '{shape}' unsupported, the choise are:\n"
            "► 'Disk' for a 2d disk,\n"
            "► 'Sphere' for a 3d sphere."
        )
    return shape


def _check_core(core):
    core = core.capitalize()
    if core not in ["Dirichlet", "Penetrable", "Neumann"]:
        raise ValueError(
            f"Core of type '{core}' unsupported, the choise are:\n"
            "► 'Dirichlet' for a Dirichlet condition on the inner radii,\n"
            "► 'Neumann' for a Neumann condition on the inner radii,\n"
            "► 'Penetrable' for a penetrable obstacle."
        )
    return core


def _check_size(_layer, nb):
    if len(_layer) is not nb:
        raise ValueError(f"{_layer} should have size {nb}.")
    return None


def _to_fct(nof_layer):
    fct_layer = []
    for nof in nof_layer:
        if np.isscalar(nof):
            fct_layer.append(Constant(nof))
        else:
            fct_layer.append(nof)
    return tuple(fct_layer)


def create_obstacle(
    shape: str,
    core: str,
    radii: ArrayLike,
    σ_layer: List[Union[int, float, complex, Callable[[np.array], np.array]]],
    ρ_layer: List[Union[int, float, complex, Callable[[np.array], np.array]]],
):

    shape = _check_shape(shape)
    core = _check_core(core)

    radii = tuple(float(r) for r in radii)
    if core.startswith("P"):
        nb_layer = len(radii)
    else:
        nb_layer = len(radii) - 1

    _check_size(σ_layer, nb_layer)
    σ_layer = _to_fct(σ_layer)

    _check_size(ρ_layer, nb_layer)
    ρ_layer = _to_fct(ρ_layer)

    return Obstacle(shape, core, radii, (σ_layer, ρ_layer))
