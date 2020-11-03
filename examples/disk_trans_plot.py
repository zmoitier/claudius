from sys import argv

import numpy as np
from context import claudius

from claudius.Helmholtz_2d import (create_problem_cst, plot_field,
                                   scattered_field, total_field)

εc = float(argv[1])
μc = float(argv[2])
k = float(argv[3])
T = float(argv[4])

prob = create_problem_cst("Penetrable", (1,), ((εc, μc),), k)

N = 128
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 5:
    U = scattered_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    type_field = "Scattered field"
else:
    U = total_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    type_field = "Total field"

plot_field(prob, X, Y, U, type_field)
