from sys import argv

import numpy as np
from context import claudius

from claudius.Helmholtz_2d import (create_problem_cst, plot_field,
                                   scattered_field, total_field)

k = float(argv[1])
T = float(argv[2])

prob = create_problem_cst("Dirichlet", (1,), (), k)

N = 128
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 3:
    U = scattered_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    type_field = "Scattered field"
else:
    U = total_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    type_field = "Total field"

plot_field(prob, X, Y, U, type_field)
