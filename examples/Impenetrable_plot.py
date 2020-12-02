from math import sqrt
from sys import argv

import numpy as np

import claudius
from claudius.plot import plot_field, plot_geometry

dim = int(argv[1])
pde = argv[2]
inn_bdy = argv[3]
k = float(argv[4])
T = float(argv[5])
which = argv[6]

N = 128
C1, C2 = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))

if dim == 2:
    from claudius.Helmholtz_2d import create_problem_cst, scattered_field, total_field

    coord = claudius.to_polar(C1, C2)

if dim == 3:
    from claudius.Helmholtz_3d import create_problem_cst, scattered_field, total_field

    plan = argv[7]
    if plan == "XY":
        coord = claudius.to_spheric(C1, C2, 0)
    if plan == "XZ":
        coord = claudius.to_spheric(C1, 0, C2)

prob = create_problem_cst(inn_bdy, (1,), (), k)
plot_geometry(prob)

sol = claudius.solve_prob(prob, radius_max=sqrt(2) * T)
if which.startswith("S"):
    U = scattered_field(sol, *coord)
    type_field = "Scattered field"

if which.startswith("T"):
    U = total_field(sol, *coord)
    type_field = "Total field"

plot_field(prob, C1, C2, U, type_field)
