from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from rot_inv_scattering import *

δ = float(argv[1])
εc = float(argv[2])
μc = float(argv[3])
k = float(argv[4])
T = 2

N = 256
x, dx = np.linspace(-T, T, num=N, retstep=True)
X, Y = np.meshgrid(x, x)

if len(argv) > 5:
    U = ann_cts.scattered_field(δ, εc, μc, k, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Scattered field"
else:
    U = ann_cts.total_field(δ, εc, μc, k, X, Y, "xy", T=np.sqrt(2) * T)
    which = "Total field"

fig, ax = plt.subplots()

plt.suptitle(
    fr"{which}: $\delta = {δ}$, $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

diskδ = plt.Circle((0, 0), δ, fill=False, ec="k", lw=2, ls="--", animated=True)
disk1 = plt.Circle((0, 0), 1, fill=False, ec="k", lw=2, ls="--", animated=True)

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
    return (im, ax.add_artist(diskδ), ax.add_artist(disk1))


def animate(i):
    global U, diskδ, disk1

    U *= expi
    im.set_array(np.real(U))
    return (im, ax.add_artist(diskδ), ax.add_artist(disk1))


anim = FuncAnimation(
    fig, animate, init_func=init, frames=nbt, blit=True, interval=20, repeat=True
)

plt.show()
