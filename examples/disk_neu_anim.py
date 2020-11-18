from sys import argv

import numpy as np
from context import claudius

from claudius.Helmholtz_2d import (anim_field, create_problem_cst,
                                   scattered_field, total_field)

k = float(argv[1])
T = float(argv[2])

prob = create_problem_cst("Neumann", (1,), (), k)

N = 256
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 3:
    U = scattered_field(prob, X, Y, "cartesian", T=np.sqrt(2) * T)
    type_field = "Scattered field"
else:
    U = total_field(prob, X, Y, "cartesian", T=np.sqrt(2) * T)
    type_field = "Total field"

anim_field(prob, X, Y, U, type_field)
