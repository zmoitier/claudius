from sys import argv

import numpy as np
from context import claudius

from claudius.Helmholtz_2d import (anim_field, create_problem_cst,
                                   scattered_field, total_field)

δ = float(argv[1])
εc = float(argv[2])
μc = float(argv[3])
k = float(argv[4])
T = float(argv[5])

prob = create_problem_cst("Penetrable", (δ, 1), ((1, 1), (εc, μc)), k)

N = 256
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 6:
    U = scattered_field(prob, X, Y, "cartesian", T=np.sqrt(2) * T)
    type_field = "Scattered field"
else:
    U = total_field(prob, X, Y, "cartesian", T=np.sqrt(2) * T)
    type_field = "Total field"

anim_field(prob, X, Y, U, type_field)
