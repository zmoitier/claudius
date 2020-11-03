from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from context import claudius as acs

from claudius.Helmholtz_2d import (create_problem_cst, scattered_field,
                                   total_field)

εc = float(argv[2])
μc = float(argv[3])
k = float(argv[4])
T = float(argv[5])

δ = 0.75
prob = create_problem_cst("Penetrable", (0.75, 1), ((1, 1), (εc, μc)), k)

N = 128
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 5:
    U = scattered_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Scattered field"
else:
    U = total_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Total field"


def my_plot(fig, ax, U, type, name):
    diskδ = plt.Circle((0, 0), δ, fill=False, ec="k", lw=2, ls="--")
    disk1 = plt.Circle((0, 0), 1, fill=False, ec="k", lw=2, ls="--")
    if type == "part":
        U_max = np.amax(np.abs(U))
        Clim = (-U_max, U_max)
    elif type == "abs":
        U_max = np.amax(U)
        Clim = (0, U_max)
    else:
        Clim = (-np.pi, np.pi)

    p = ax.pcolormesh(X, Y, np.real(U), shading="gouraud", cmap=Cmap[type], clim=Clim)
    ax.add_artist(diskδ)
    ax.add_artist(disk1)
    fig.colorbar(p, ax=ax)
    ax.set_title(name)


fig, ax = plt.subplots(2, 2)

Cmap = {"part": "RdBu_r", "abs": "viridis", "arg": "twilight_shifted_r"}
my_plot(ax[0, 0], np.real(U), "part", "Real part")
my_plot(ax[0, 1], np.imag(U), "part", "Imaginary part")
my_plot(ax[1, 0], np.abs(U), "abs", "Modulus")
my_plot(ax[1, 1], np.angle(U), "arg", "Argument")

plt.suptitle(
    fr"{which}: $\delta = {δ}$, $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

plt.tight_layout()

plt.show()
