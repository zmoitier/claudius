import matplotlib.pyplot as plt
import numpy as np

import claudius as acs
from claudius.Helmholtz_2d import CD_cst, CD_cst_der

dim = 2
pde = "H"
type = "N"
radii = (0.5, 0.75, 1)
k = 5
if type.startswith("P"):
    εμc = (
        (lambda r: np.ones_like(r), lambda r: np.ones_like(r)),
        (lambda r: 2 * np.ones_like(r), lambda r: np.ones_like(r)),
        (lambda r: 1.5 * np.ones_like(r), lambda r: -np.ones_like(r)),
    )

    fun = (CD_cst(1, 1, k)[0], CD_cst(1, 1, k), CD_cst(1, 1, k))
    fun_der = (CD_cst_der(1, 1, k)[0], CD_cst_der(1, 1, k), CD_cst_der(1, 1, k))
else:
    εμc = (
        (lambda r: 2 * np.ones_like(r), lambda r: np.ones_like(r)),
        (lambda r: 1.5 * np.ones_like(r), lambda r: -np.ones_like(r)),
    )
    fun = (CD_cst(1, 1, k), CD_cst(1, 1, k))
    fun_der = (CD_cst_der(1, 1, k), CD_cst_der(1, 1, k))

M = acs.M_trunc_2d(k, 2)
prob = acs.create_probem(dim, pde, type, radii, εμc, k, fun, fun_der)

acs.plot_geometry(prob)

"""
sol = acs.solve_prob(prob, M)

r = np.linspace(0.25, 1.25, num=16)
θ = np.linspace(0, 2 * np.pi, num=16, endpoint=False)
R, Θ = np.meshgrid(r, θ)

us = acs.tt_field(sol, R, Θ)
plt.imshow(np.real(us))
plt.colorbar()

plt.show()
"""
