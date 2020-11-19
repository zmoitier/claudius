import math as ma
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import claudius as acs
from claudius.Helmholtz_3d import fun_cst, fun_cst_der

dim = 3
pde = "H"
inn_bdy = "P"
N = int(argv[1])
k = 5

if N == 0:
    radii = (1,)
else:
    radii = tuple(np.linspace(1, 1.5, num=N + 1))

if inn_bdy.startswith("P"):
    εμc = tuple((1, 1) for n in range(N + 1))
    fun = (fun_cst(1, 1, k)[0], *(fun_cst(1, 1, k) for n in range(N)))
    fun_der = (fun_cst_der(1, 1, k)[0], *(fun_cst_der(1, 1, k) for n in range(N)))
else:
    εμc = tuple((1, 1) for n in range(N))
    fun = tuple(fun_cst(1, 1, k) for n in range(N))
    fun_der = tuple(fun_cst_der(1, 1, k) for n in range(N))

prob = acs.create_probem(dim, pde, inn_bdy, radii, εμc, k, fun, fun_der)
print(prob.eps_mu)

M = acs.trunc_H3d(k, 2)

sol = acs.solve_prob(prob, M)
print(sol)
