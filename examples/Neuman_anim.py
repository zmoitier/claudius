from sys import argv

import numpy as np
from context import claudius

from claudius.plot import anim_field

dim = int(argv[1])
pde = argv[2]
k = float(argv[3])
T = float(argv[4])
which = argv[5]

N = 256
C1, C2 = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))

if dim == 2:
    from claudius.Helmholtz_2d import (create_problem_cst, scattered_field,
                                       total_field)

    coord = (C1, C2)

if dim == 3:
    from claudius.Helmholtz_3d import (create_problem_cst, scattered_field,
                                       total_field)

    plan = argv[6]
    if plan == "XY":
        coord = C1, C2, np.array([0])
    if plan == "XZ":
        coord = C1, np.array([0]), C2

prob = create_problem_cst("Neumann", (1,), (), k)

if which.startswith("S"):
    U = scattered_field(prob, *coord, "cartesian")
    type_field = "Scattered field"

if which.startswith("T"):
    U = total_field(prob, *coord, "cartesian")
    type_field = "Total field"

anim_field(prob, C1, C2, U, type_field)
