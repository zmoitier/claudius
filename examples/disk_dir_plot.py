from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from acswave.Helmholtz_2d import CD_cst, CD_cst_der
from context import acswave as acs
from scipy.special import hankel1, jv

k = float(argv[1])
T = 2

dim = 2
pde = "H"
type = "D"
radii = (0.5, 1)
εμc = ((lambda r: np.ones_like(r), lambda r: np.ones_like(r)),)
fun = (CD_cst(1, 1, k),)
fun_der = (CD_cst_der(1, 1, k),)

M = acs.M_trunc_2d(k, T)
prob = acs.create_probem(dim, pde, type, radii, εμc, k, fun, fun_der)
sol = acs.solve_prob(prob, M)

N = 128
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
R, Θ = np.hypot(X, Y), np.arctan2(Y, X)
if len(argv) > 2:
    U = us = acs.sc_field(sol, R, Θ)
    which = "Scattered field"
else:
    U = acs.tt_field(sol, R, Θ)
    which = "Total field"

Cmap = {"part": "RdBu_r", "abs": "viridis", "arg": "twilight_shifted_r"}


def my_plot(ax, U, type, name):
    disk = plt.Circle((0, 0), 1, fc=(0.75, 0.75, 0.75), ec="k", lw=2)
    if type == "part":
        U_max = np.amax(np.abs(U))
        Clim = (-U_max, U_max)
    elif type == "abs":
        U_max = np.amax(U)
        Clim = (0, U_max)
    else:
        Clim = (-np.pi, np.pi)

    p = ax.pcolormesh(X, Y, np.real(U), shading="gouraud", cmap=Cmap[type], clim=Clim)
    # ax.add_artist(disk)
    ax.axis("equal")
    plt.colorbar(p, ax=ax)
    ax.set_title(name)


fig, ax = plt.subplots(2, 2)
my_plot(ax[0, 0], np.real(U), "part", "Real part")
my_plot(ax[0, 1], np.imag(U), "part", "Imaginary part")
my_plot(ax[1, 0], np.abs(U), "abs", "Modulus")
my_plot(ax[1, 1], np.angle(U), "arg", "Argument")

plt.suptitle(fr"{which}: $k = {k}$")

plt.tight_layout()

plt.show()
