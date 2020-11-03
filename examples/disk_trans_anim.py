from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from context import claudius as acs
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

from claudius.Helmholtz_2d import (create_problem_cst, scattered_field,
                                   total_field)

εc = float(argv[1])
μc = float(argv[2])
k = float(argv[3])
T = float(argv[3])

prob = create_problem_cst("Penetrable", (1,), ((εc, μc),), k)

N = 256
x, dx = np.linspace(-T, T, num=N, retstep=True)
X, Y = np.meshgrid(x, x)

if len(argv) > 5:
    U = scattered_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Scattered field"
else:
    U = total_field(prob, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Total field"

fig, ax = plt.subplots()

plt.title(
    fr"{which} with $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

disk = plt.Circle((0, 0), 1, fill=False, ec="k", lw=2, ls="--", animated=True)

UaM = np.amax(np.abs(U))
im = ax.imshow(
    np.real(U),
    interpolation="none",
    cmap="RdBu_r",
    origin="lower",
    aspect="equal",
    extent=(-T - dx / 2, T + dx / 2, -T - dx / 2, T + dx / 2),
    animated=True,
    vmin=-UaM,
    vmax=UaM,
)
cbar = plt.colorbar(im, ax=ax)

nbt = 110
dt = 2 * np.pi / (k * nbt)
expi = np.exp(-1j * k * dt)


def init():
    return (im, ax.add_artist(disk))


def animate(i):
    global U, disk

    U *= expi
    im.set_array(np.real(U))
    return (im, ax.add_artist(disk))


anim = FuncAnimation(
    fig, animate, init_func=init, frames=nbt, blit=True, interval=20, repeat=True
)

plt.show()
