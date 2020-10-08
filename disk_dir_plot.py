from sys import argv

import matplotlib.pyplot as plt
import numpy as np

from rot_inv_scattering import *

k = float(argv[1])
T = 2
sol = disk_dir.solution(k, T)

N = 128
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 2:
    U = scattered_field(k, sol, X, Y, "xy")
    which = "Scattered field"
else:
    U = total_field(k, sol, X, Y, "xy")
    which = "Total field"

Cmap = {"part": "RdBu_r", "abs": "viridis", "arg": "twilight_shifted_r"}


def my_plot(ax, U, type, name):
    disk = plt.Circle((0, 0), 1, color="k", alpha=0.25)
    if type == "part":
        U_max = np.amax(np.abs(U))
        Clim = (-U_max, U_max)
    elif type == "abs":
        U_max = np.amax(U)
        Clim = (0, U_max)
    else:
        Clim = (-np.pi, np.pi)

    p = ax.pcolormesh(X, Y, np.real(U), shading="gouraud", cmap=Cmap[type], clim=Clim)
    ax.add_artist(disk)
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
