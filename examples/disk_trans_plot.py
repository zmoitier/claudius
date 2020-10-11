from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from context import accoster as acs

εc = float(argv[1])
μc = float(argv[2])
k = float(argv[3])
T = 2

N = 64
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 4:
    U = acs.disk_trans.scattered_field(εc, μc, k, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Scattered field"
else:
    U = acs.disk_trans.total_field(εc, μc, k, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Total field"

Cmap = {"part": "RdBu_r", "abs": "viridis", "arg": "twilight_shifted_r"}


def my_plot(ax, U, type, name):
    disk = plt.Circle((0, 0), 1, fill=False, ec="k", lw=2, ls="--")
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

plt.suptitle(
    fr"{which}: $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

plt.tight_layout()

plt.show()
