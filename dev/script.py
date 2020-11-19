from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from context import claudius

from claudius.Helmholtz_3d import fun_cst, fun_cst_der

dim = 2
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

prob = claudius.create_probem(dim, pde, inn_bdy, radii, εμc, k, fun, fun_der)
M = claudius.trunc_H3d(k, 2)
sol = claudius.solve_prob(prob, M)

x = np.linspace(-2, 2, num=64)
z = np.linspace(-2, 2, num=64)
X, Z = np.meshgrid(x, z)
R, T, P = claudius.to_spheric(X, np.array([0]), Z, "cartesian")

U = claudius.tt_field(sol, R, T)
plt.imshow(np.abs(U), extent=(-2, 2, -2, 2))

t = np.linspace(0, 2 * np.pi, num=64)
co, si = np.cos(t), np.sin(t)
for ρ in radii:
    plt.plot(ρ * co, ρ * si, "k")

plt.colorbar()

plt.show()
