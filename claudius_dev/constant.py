""" Constant class """
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class Constant:
    """
    Dataclass to describe the constant function _ ↦ value.

    Attributes
    ----------
    value : Union[int, float, complex]
        Scalar value
    """

    value: Union[int, float, complex]

    def __call__(self, r):
        return np.full_like(r, self.value)

    def __repr__(self):
        return f"_ ↦ {self.value}"
