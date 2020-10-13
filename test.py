import matplotlib.pyplot as plt
import numpy as np

import acswave as acs

dim = 2
pde = "H"
type = "P"
radii = (0.5, 0.75, 1)
εμc = (
    (lambda r: np.ones_like(r), lambda r: np.ones_like(r)),
    (lambda r: np.ones_like(r), lambda r: np.ones_like(r)),
    (lambda r: np.ones_like(r), lambda r: np.ones_like(r)),
)
k = 5
fun = (acs.CD_cst(1, 1, k)[0], acs.CD_cst(1, 1, k), acs.CD_cst(1, 1, k))
fun_der = (acs.CD_cst_der(1, 1, k)[0], acs.CD_cst_der(1, 1, k), acs.CD_cst_der(1, 1, k))

M = acs.M_trunc_2d(k, 2)
prob = acs.create_probem(dim, pde, type, radii, εμc, k, fun, fun_der)
sol = acs.solve_prob(prob, M)
us = acs.sc_field_2d(sol, 1, 0)
print(us)
