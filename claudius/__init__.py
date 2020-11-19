"""isort:skip_file"""
from .core import Problem, Solution, create_probem
from .utils import (
    abs2,
    εμ_to_n,
    εμ_to_η,
    trunc_H2d,
    trunc_H3d,
    trunc_None,
)
from .coords import to_polar, to_spheric

from .solve_prob import solve_prob

from .calc_field import sc_field, tt_field
from .far_field import far_field

__version__ = "1.1.0"
