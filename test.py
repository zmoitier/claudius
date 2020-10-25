from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import claudius as acs
from claudius.Helmholtz_2d import CD_cst, CD_cst_der

dim = 2
pde = "H"
inn_bdy = "P"
N = int(argv[1])
k = 5

if N == 0:
    radii = (1,)
else:
    radii = tuple(np.linspace(0.5, 1, num=N + 1))

if inn_bdy.startswith("P"):
    εμc = tuple(
        (lambda r: np.ones_like(r), lambda r: np.ones_like(r)) for n in range(N + 1)
    )
    fun = (CD_cst(1, 1, k)[0], *(CD_cst(1, 1, k) for n in range(N)))
    fun_der = (CD_cst_der(1, 1, k)[0], *(CD_cst_der(1, 1, k) for n in range(N)))
else:
    εμc = tuple(
        (lambda r: np.ones_like(r), lambda r: np.ones_like(r)) for n in range(N)
    )
    fun = tuple(CD_cst(1, 1, k) for n in range(N))
    fun_der = tuple(CD_cst_der(1, 1, k) for n in range(N))

M = acs.trunc_H2d(k, 2)
prob = acs.create_probem(dim, pde, inn_bdy, radii, εμc, k, fun, fun_der)

sol = acs.solve_prob(prob, M)
for v in range(np.size(sol.coeff, 1)):
    print(sol.coeff[:, v])
