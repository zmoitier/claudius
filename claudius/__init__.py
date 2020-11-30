"""Analytic computations for scattering"""
from .coordinates import to_polar, to_spheric
from .core import Problem, Solution, create_probem
from .solve_prob import solve_prob
from .utils import abs2, εμ_to_n, εμ_to_η

__version__ = "1.1.1"
