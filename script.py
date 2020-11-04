from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import claudius as acs
from claudius.Helmholtz_2d import far_field, fun_cst, fun_cst_der, plot_far_field

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

θ = np.linspace(0, 2 * np.pi, num=128)
ff = far_field(prob, θ)

plot_far_field(θ, ff)
