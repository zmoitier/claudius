from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import rot_inv_scattering.disk as disk

εc = float(argv[1])
μc = float(argv[2])
k = float(argv[3])

N = 256
T = 2
x, dx = np.linspace(-T, T, num=N, retstep=True)
X, Y = np.meshgrid(x, x)

fig = plt.figure()

if len(argv) > 4:
    U = disk.scattered_field(εc, μc, k, T, X, Y, "xy")
    which = "Scattered field"
else:
    U = disk.total_field(εc, μc, k, T, X, Y, "xy")
    which = "Total field"

plt.title(
    fr"{which}: $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

UaM = np.amax(np.abs(U))
im = plt.imshow(
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
plt.xticks()
plt.yticks()
cbar = plt.colorbar()
cbar.ax.tick_params()

nbt = 110
dt = 2 * np.pi / (k * nbt)
expi = np.exp(-1j * k * dt)


def animate(i):
    global U

    U *= expi
    im.set_array(np.real(U))
    return (im,)


anim = FuncAnimation(fig, animate, frames=nbt, blit=True, interval=20, repeat=True)

plt.show()
