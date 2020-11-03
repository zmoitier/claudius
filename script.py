from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import claudius as acs
from claudius.Helmholtz_2d import create_problem_cst, fun_cst, fun_cst_der

dim = 2
pde = "H"
inn_bdy = "N"
N = int(argv[1])
k = 5

if N == 0:
    radii = (1,)
else:
    radii = tuple(np.linspace(1, 1.5, num=N + 1))

if inn_bdy.startswith("P"):
    εμc = tuple(
        (lambda r: np.ones_like(r), lambda r: np.ones_like(r)) for n in range(N + 1)
    )
    fun = (fun_cst(1, 1, k)[0], *(fun_cst(1, 1, k) for n in range(N)))
    fun_der = (fun_cst_der(1, 1, k)[0], *(fun_cst_der(1, 1, k) for n in range(N)))
else:
    εμc = tuple(
        (lambda r: np.ones_like(r), lambda r: np.ones_like(r)) for n in range(N)
    )
    fun = tuple(fun_cst(1, 1, k) for n in range(N))
    fun_der = tuple(fun_cst_der(1, 1, k) for n in range(N))

M = acs.trunc_H2d(k, 2)
prob = acs.create_probem(dim, pde, inn_bdy, radii, εμc, k, fun, fun_der)

sol = acs.solve_prob(prob, M)

nb = 64
H = 2
x = np.linspace(-H, H, num=nb)
X, Y = np.meshgrid(x, x)
R, Θ = np.hypot(X, Y), np.arctan2(Y, X)

U = acs.sc_field(sol, R, Θ)

plt.pcolormesh(X, Y, np.abs(U), shading="nearest")
plt.colorbar()

plt.show()
